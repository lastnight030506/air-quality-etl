"""OpenAQ v3 extract.

OpenAQ is keyed: every request carries `X-API-Key`, and an unauthenticated call
is a guaranteed 401. We fail before the request instead of spending it.

The hourly endpoint stamps each row with the END of its period, so `hour_utc`
comes from align.openaq_hour_start_from_period. Absent hours are not returned at
all; gaps.py rebuilds the spine.
"""

from __future__ import annotations

import os
import time
from datetime import UTC, datetime

import httpx
import polars as pl

from . import align, schema
from .config import (
    CITIES,
    OPENAQ_BASE,
    OPENAQ_MAX_PER_PAGE,
    OPENAQ_TIMEOUT_S,
    PARAMETERS,
    SEARCH_RADIUS_M,
    City,
)

_MISSING_KEY = (
    "OPENAQ_API_KEY is not set. Register a free key at "
    "https://explore.openaq.org/register and export it (or add it as a repo "
    "secret named OPENAQ_API_KEY)."
)

_LOCATION_PAGE = 100
_MAX_LOCATION_PAGES = 5
_MAX_429_RETRIES = 3
_PAGE_SLEEP_S = 0.05
_DEFAULT_RESET_S = 60

EXTRACT_COLUMNS: tuple[str, ...] = (
    "city",
    "station_id",
    "station_name",
    "sensor_id",
    "parameter",
    "hour_utc",
    "value_raw",
    "unit",
    "coverage_pct",
)

EXTRACT_SCHEMA: dict[str, pl.DataType] = {
    name: schema.CURATED_SCHEMA[name] for name in EXTRACT_COLUMNS
}


def _api_key() -> str:
    key = os.environ.get("OPENAQ_API_KEY", "").strip()
    if not key:
        raise RuntimeError(_MISSING_KEY)
    return key


def _iso(stamp: datetime) -> str:
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=UTC)
    return stamp.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_utc(stamp: str | None) -> datetime | None:
    if not stamp:
        return None
    return datetime.fromisoformat(stamp.replace("Z", "+00:00"))


def _reset_seconds(response: httpx.Response) -> int:
    try:
        return int(response.headers.get("x-ratelimit-reset", _DEFAULT_RESET_S))
    except ValueError:  # header may be an HTTP date rather than seconds
        return _DEFAULT_RESET_S


def _request(client: httpx.Client, url: str, params: dict) -> dict:
    headers = {"X-API-Key": _api_key()}
    response = client.get(url, params=params, headers=headers)
    for _ in range(_MAX_429_RETRIES):
        if response.status_code != 429:
            break
        time.sleep(_reset_seconds(response))
        response = client.get(url, params=params, headers=headers)
    if response.status_code == 401:
        raise RuntimeError(_MISSING_KEY)
    response.raise_for_status()
    return response.json()


def fetch_locations(client: httpx.Client, city: City) -> list[dict]:
    params = {
        "coordinates": f"{city.latitude},{city.longitude}",
        "radius": SEARCH_RADIUS_M,
        # The API types this as array<int>; httpx repeats the key for a list.
        # A comma-joined string is silently ignored, which returns every
        # location in range and then filters to nothing downstream.
        "parameters_id": sorted(PARAMETERS),
        "limit": _LOCATION_PAGE,
    }
    locations: list[dict] = []
    for page in range(1, _MAX_LOCATION_PAGES + 1):
        if page > 1:
            time.sleep(_PAGE_SLEEP_S)
        params["page"] = page
        payload = _request(client, f"{OPENAQ_BASE}/locations", params)
        batch = payload.get("results") or []
        locations.extend(batch)
        found = (payload.get("meta") or {}).get("found")
        if len(batch) < _LOCATION_PAGE:
            break
        # `found` is an int, or a string like ">1000" that we cannot compare.
        if isinstance(found, int) and len(locations) >= found:
            break
    return locations


def fetch_hours(
    client: httpx.Client, sensor_id: int, start: datetime, end: datetime
) -> list[dict]:
    params = {
        "datetime_from": _iso(start),
        "datetime_to": _iso(end),
        "limit": OPENAQ_MAX_PER_PAGE,
    }
    results: list[dict] = []
    page = 1
    while True:
        if page > 1:
            time.sleep(_PAGE_SLEEP_S)
        params["page"] = page
        payload = _request(
            client, f"{OPENAQ_BASE}/sensors/{sensor_id}/hours", params
        )
        batch = payload.get("results") or []
        results.extend(batch)
        if len(batch) < OPENAQ_MAX_PER_PAGE:
            break
        page += 1
    return results


def _target_sensors(location: dict) -> list[tuple[int, str]]:
    sensors = []
    for sensor in location.get("sensors") or []:
        parameter_id = (sensor.get("parameter") or {}).get("id")
        if parameter_id in PARAMETERS:
            sensors.append((sensor.get("id"), PARAMETERS[parameter_id]))
    return sensors


def _curated_row(
    city: City, location: dict, sensor_id: int, parameter: str, reading: dict
) -> dict | None:
    value = reading.get("value")
    if value is None:
        return None
    period = reading.get("period") or {}
    coverage = reading.get("coverage") or {}
    # The end stamp lives in `period`; fall back to `coverage` when it is absent
    # so a reading without a period still lands in the right hour.
    to_stamp = _parse_utc((period.get("datetimeTo") or {}).get("utc")) or _parse_utc(
        (coverage.get("datetimeTo") or {}).get("utc")
    )
    if to_stamp is None:
        return None
    from_stamp = _parse_utc((period.get("datetimeFrom") or {}).get("utc"))
    return {
        "city": city.key,
        "station_id": location.get("id"),
        "station_name": location.get("name"),
        "sensor_id": sensor_id,
        "parameter": parameter,
        "hour_utc": align.openaq_hour_start_from_period(from_stamp, to_stamp),
        "value_raw": value,
        "unit": (reading.get("parameter") or {}).get("units"),
        "coverage_pct": coverage.get("percentComplete"),
    }


def _last_seen(location: dict) -> str:
    return (location.get("datetimeLast") or {}).get("utc") or ""


def fetch_city(
    client: httpx.Client,
    city: City,
    start: datetime,
    end: datetime,
    *,
    max_stations: int = 5,
) -> pl.DataFrame:
    all_locations = fetch_locations(client, city)
    stations = [
        (location, _target_sensors(location))
        for location in all_locations
        if not location.get("isMobile")
    ]
    stations = [pair for pair in stations if pair[1]]
    # Newest first, and only stations that reported inside the window. OpenAQ
    # keeps retired stations in its catalogue: sorting by id picked five that
    # had been offline for years and produced a silent zero-row run while a
    # station reporting this hour sat further down the list.
    since = start.strftime("%Y-%m-%dT%H:%M:%SZ")
    stations.sort(key=lambda pair: _last_seen(pair[0]), reverse=True)
    live = [pair for pair in stations if _last_seen(pair[0]) >= since]
    # Per-city visibility: a silent zero here is the difference between "the
    # city has no stations" and "our query is wrong", and the run log is the
    # only place that distinction can be recovered later.
    print(
        f"  {city.key:8} locations={len(all_locations)}"
        f" with_target_sensors={len(stations)} live_in_window={len(live)}"
        f" newest={_last_seen(stations[0][0]) if stations else 'none'}"
        f" taking={min(len(live), max_stations)}",
        flush=True,
    )

    rows = []
    for location, sensors in live[:max_stations]:
        for sensor_id, parameter in sensors:
            readings = fetch_hours(client, sensor_id, start, end)
            for reading in readings:
                row = _curated_row(city, location, sensor_id, parameter, reading)
                if row is not None:
                    rows.append(row)
    return pl.DataFrame(rows, schema=EXTRACT_SCHEMA)


def fetch_all(
    start: datetime, end: datetime, *, cities: tuple[City, ...] | None = None
) -> pl.DataFrame:
    frames = []
    with httpx.Client(timeout=OPENAQ_TIMEOUT_S) as client:
        for city in CITIES if cities is None else cities:
            frames.append(fetch_city(client, city, start, end))
    if not frames:
        return pl.DataFrame(schema=EXTRACT_SCHEMA)
    return pl.concat(frames, how="vertical")
