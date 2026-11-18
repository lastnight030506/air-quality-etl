"""Open-Meteo extract.

Keyless. Two hosts serve the same payload shape: `archive-api` for anything
older than yesterday, and `forecast` with `past_days` for recent history.

Open-Meteo returns local wall-clock strings for the requested timezone, and each
stamp is the START of its hour, so hour_utc comes straight from
align.openmeteo_local_to_utc.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from time import sleep
from zoneinfo import ZoneInfo

import httpx
import polars as pl

from . import align, schema
from .config import CITIES, City

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
HOURLY = (
    "temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation"
)
TIMEZONE = "Asia/Ho_Chi_Minh"
PAST_DAYS = 2
_ATTEMPTS = 3
_BACKOFF_S = 1.0
TIMEOUT_S = 30.0


def _get_with_retry(client: httpx.Client, url: str, params: dict) -> httpx.Response:
    """Retry the TLS handshake, which is what actually fails on CI runners.

    Observed twice in GitHub Actions: ConnectTimeout on the handshake, while the
    same host answers in ~1 s locally. Weather is enrichment, but silently
    shipping nulls whenever a runner is unlucky is worse than a short retry.
    """
    last: Exception | None = None
    for attempt in range(_ATTEMPTS):
        try:
            response = client.get(url, params=params)
            response.raise_for_status()
            return response
        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout) as exc:
            last = exc
            if attempt < _ATTEMPTS - 1:
                sleep(_BACKOFF_S * (attempt + 1))
    assert last is not None
    raise last


def _fetch(
    client: httpx.Client,
    city: City,
    start: date,
    end: date,
    *,
    archive: bool,
) -> pl.DataFrame:
    params = {
        "latitude": city.latitude,
        "longitude": city.longitude,
        "hourly": HOURLY,
        "timezone": TIMEZONE,
    }
    # `past_days` is mutually exclusive with start_date/end_date (HTTP 400), so
    # the recent branch asks for the trailing window and _clip trims it.
    if archive:
        params["start_date"] = start.isoformat()
        params["end_date"] = end.isoformat()
    else:
        params["past_days"] = max(PAST_DAYS, min(92, (date.today() - start).days + 2))
    response = _get_with_retry(client, ARCHIVE_URL if archive else FORECAST_URL, params)
    hourly = response.json().get("hourly") or {}
    stamps = [align.openmeteo_local_to_utc(s) for s in hourly.get("time") or []]
    n = len(stamps)

    def series(name: str) -> list:
        # A missing variable must not shorten the frame; leave it null instead.
        return hourly.get(name) or [None] * n

    return pl.DataFrame(
        {
            "city": [city.key] * n,
            "hour_utc": stamps,
            "temperature_2m": series("temperature_2m"),
            "relative_humidity_2m": series("relative_humidity_2m"),
            "wind_speed_10m": series("wind_speed_10m"),
            "precipitation": series("precipitation"),
        },
        schema=schema.WEATHER_SCHEMA,
    )


def _clip(df: pl.DataFrame, start: date, end: date) -> pl.DataFrame:
    tz = ZoneInfo(TIMEZONE)
    lower = datetime.combine(start, time(), tzinfo=tz)
    upper = datetime.combine(end + timedelta(days=1), time(), tzinfo=tz)
    return df.filter(
        (pl.col("hour_utc") >= lower.astimezone(ZoneInfo("UTC")))
        & (pl.col("hour_utc") < upper.astimezone(ZoneInfo("UTC")))
    )


def fetch_city(
    client: httpx.Client, city: City, start: date, end: date
) -> pl.DataFrame:
    today = date.today()
    recent = today - timedelta(days=1)
    frames = []
    if start < recent:
        frames.append(
            _fetch(client, city, start, min(end, recent - timedelta(days=1)), archive=True)
        )
    if end >= recent:
        frames.append(
            _fetch(client, city, max(start, recent), end, archive=False)
        )
    if not frames:
        return schema.empty_weather()
    df = pl.concat(frames, how="vertical").unique(subset=["hour_utc"], keep="first")
    return _clip(df, start, end).sort("hour_utc")


def fetch_all(
    start: date, end: date, *, cities: tuple[City, ...] | None = None
) -> pl.DataFrame:
    frames = []
    with httpx.Client(timeout=TIMEOUT_S) as client:
        for city in CITIES if cities is None else cities:
            frames.append(fetch_city(client, city, start, end))
    if not frames:
        return schema.empty_weather()
    return pl.concat(frames, how="vertical")
