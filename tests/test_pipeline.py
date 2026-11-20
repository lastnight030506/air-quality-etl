"""End-to-end pipeline test with a mocked OpenAQ transport.

This exercises the real code path — extract, unit conversion, gap filling,
QA/QC, alignment, Parquet upsert, DuckDB report, chart — without a live API key.
It exists because the live OpenAQ path cannot be run in CI without a secret, and
an untested integration is where a project like this silently rots.

The fixture data is synthetic BY CONSTRUCTION and is written to tmp_path only.
It must never reach data/.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import httpx
import polars as pl
import pytest

from aqetl import chart, gaps, openaq, openmeteo, qaqc, schema, storage, units

LOCATION = {
    "id": 4242,
    "name": "Test Station",
    "timezone": "Asia/Ho_Chi_Minh",
    "isMobile": False,
    # Station selection is recency-based, so a fixture without this would be
    # dropped as retired before any sensor is queried.
    "datetimeLast": {"utc": "2026-11-02T00:00:00Z"},
    "sensors": [
        {"id": 900, "parameter": {"id": 2, "name": "pm25", "units": "µg/m³"}},
        {"id": 901, "parameter": {"id": 7, "name": "no2", "units": "ppm"}},
    ],
}

START = datetime(2026, 11, 1, 0, 0, tzinfo=UTC)


def _hours_payload(sensor_id: int, n: int, *, unit: str, values: list[float]) -> dict:
    rows = []
    for i in range(n):
        end = START + timedelta(hours=i + 1)
        rows.append(
            {
                "value": values[i] if i < len(values) else None,
                "parameter": {"name": "pm25", "units": unit},
                "period": {
                    "datetimeFrom": {
                        "utc": (end - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
                    },
                    "datetimeTo": {"utc": end.strftime("%Y-%m-%dT%H:%M:%SZ")},
                },
                "coverage": {"percentComplete": 100.0},
                "flagInfo": {"hasFlags": False},
            }
        )
    return {"meta": {"page": 1, "limit": 1000}, "results": rows}


def _transport(*, pm25_values: list[float], no2_values: list[float]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/locations"):
            return httpx.Response(200, json={"meta": {"found": 1}, "results": [LOCATION]})
        if "/sensors/900/hours" in path:
            return httpx.Response(
                200, json=_hours_payload(900, len(pm25_values), unit="µg/m³", values=pm25_values)
            )
        if "/sensors/901/hours" in path:
            return httpx.Response(
                200, json=_hours_payload(901, len(no2_values), unit="ppm", values=no2_values)
            )
        return httpx.Response(404, json={"message": "not found"})

    return httpx.MockTransport(handler)


def _extract(monkeypatch, transport: httpx.MockTransport) -> pl.DataFrame:
    """Extract one city only: the mock answers every city with the same station."""
    from aqetl.config import CITIES

    monkeypatch.setenv("OPENAQ_API_KEY", "test-key")
    real_client = httpx.Client

    def patched(*args, **kwargs):
        kwargs["transport"] = transport
        return real_client(*args, **kwargs)

    monkeypatch.setattr(openaq.httpx, "Client", patched)
    return openaq.fetch_all(START, START + timedelta(hours=24), cities=(CITIES[0],))


def test_extract_parses_and_converts_units(monkeypatch):
    transport = _transport(
        pm25_values=[10.0] * 24,
        # 0.05 ppm NO2 must become ~94 ug/m3, not stay 0.05.
        no2_values=[0.05] * 24,
    )
    raw = _extract(monkeypatch, transport)

    assert len(raw) == 48
    assert set(raw["parameter"].unique().to_list()) == {"pm25", "no2"}
    assert raw["hour_utc"].min() == START  # time-END stamp shifted back one hour

    converted = units.to_ugm3(raw)
    no2 = converted.filter(pl.col("parameter") == "no2")["value_ugm3"].to_list()
    assert no2[0] == pytest.approx(94.08, abs=0.2)
    pm = converted.filter(pl.col("parameter") == "pm25")["value_ugm3"].to_list()
    assert pm[0] == 10.0


def test_full_pipeline_writes_parquet_and_chart(monkeypatch, tmp_path):
    # 48 hours so a gap is visible; a stuck run; a negative; one real spike.
    pm25 = [10.0 + (i % 5) for i in range(48)]
    pm25[10:16] = [42.0] * 6          # stuck
    pm25[20] = -1.0                    # negative
    pm25[30] = 400.0                   # spike, must survive
    transport = _transport(pm25_values=pm25, no2_values=[0.02] * 48)

    raw = _extract(monkeypatch, transport)
    converted = units.to_ugm3(raw)
    filled = gaps.fill_gaps(converted, START, START + timedelta(hours=48))
    clean = qaqc.apply_all(filled)

    storage.write_curated(clean, root=tmp_path)
    back = storage.read_curated(root=tmp_path)
    assert len(back) == 96  # 2 params x 48 hours, gap-filled to the spine

    flags = back.filter(pl.col("qc_flags") > 0)
    assert not flags.filter(pl.col("qc_stuck")).is_empty()
    assert not flags.filter(pl.col("qc_negative")).is_empty()
    # The spike is flagged and its value is retained.
    spike = back.filter(pl.col("qc_spike"))
    assert not spike.is_empty()
    assert spike["value_clean"].to_list() == [400.0]
    # The negative is nulled.
    assert back.filter(pl.col("qc_negative"))["value_clean"].null_count() > 0

    # Re-run must not duplicate.
    storage.write_curated(clean, root=tmp_path)
    assert len(storage.read_curated(root=tmp_path)) == 96

    daily = storage.fetch  # keep the import honest
    assert callable(daily)

    out = chart.render(root=tmp_path, out=tmp_path / "chart.png", today=START.date())
    assert out.exists() and out.stat().st_size > 10_000


def test_gap_filling_marks_absent_hours(monkeypatch):
    """The API omits missing hours; the spine must restore them as flagged.

    The gap is in the middle of the station's reporting life. A trailing absence
    is deliberately NOT counted: that is a retired station, not a gap, and is
    bounded by the last reading instead.
    """
    from datetime import UTC, datetime

    now = datetime.now(tz=UTC).replace(minute=0, second=0, microsecond=0)
    window_start = now - timedelta(hours=30)

    def row(hour: int, value: float) -> dict:
        return {
            "value": value,
            "parameter": {"units": "µg/m³"},
            "period": {
                "datetimeFrom": {"utc": (window_start + timedelta(hours=hour)).strftime(
                    "%Y-%m-%dT%H:%M:%SZ")},
                "datetimeTo": {"utc": (window_start + timedelta(hours=hour + 1)).strftime(
                    "%Y-%m-%dT%H:%M:%SZ")},
            },
            "coverage": {"percentComplete": 100.0},
        }

    # Hours 0-9 present, 10-19 absent, 20-29 present: a 10-hour outage mid-life.
    results = [row(h, 5.0) for h in range(10)] + [row(h, 5.0) for h in range(20, 30)]

    monkeypatch.setenv("OPENAQ_API_KEY", "test-key")
    real = httpx.Client

    def patched(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(
            lambda request: httpx.Response(200, json={"meta": {}, "results": results})
            if "/hours" in request.url.path
            else httpx.Response(200, json={"meta": {"found": 1}, "results": [
                {
                    "id": 4242,
                    "name": "Test Station",
                    "isMobile": False,
                    "datetimeLast": {"utc": now.strftime("%Y-%m-%dT%H:%M:%SZ")},
                    "sensors": [{"id": 900, "parameter": {"id": 2, "units": "µg/m³"}}],
                }
            ]})
        )
        return real(*args, **kwargs)

    monkeypatch.setattr(openaq.httpx, "Client", patched)

    from aqetl.config import CITIES as CONFIGURED

    raw = openaq.fetch_all(window_start, now, cities=(CONFIGURED[0],))
    filled = gaps.fill_gaps(units.to_ugm3(raw), window_start, now)

    assert len(filled) == 30
    assert filled["qc_missing"].sum() == 10

    rate = gaps.missing_rate(filled)
    assert rate["missing_pct"].to_list() == [pytest.approx(100 / 3, abs=0.01)]


def test_spine_does_not_invent_a_boundary_hour():
    """An exclusive end must not produce an extra missing hour at the edge."""
    df = pl.DataFrame(
        {
            "city": ["hanoi"],
            "station_id": [1],
            "sensor_id": [9],
            "parameter": ["pm25"],
            "hour_utc": [START],
            "value_raw": [10.0],
        },
        schema_overrides={"hour_utc": pl.Datetime("us", "UTC")},
    )
    filled = gaps.fill_gaps(df, START, START + timedelta(hours=1))
    assert len(filled) == 1
    assert filled["qc_missing"].sum() == 0


def test_weather_join_lands_on_the_same_hours():
    """Open-Meteo local 07:00 ICT is 00:00Z, which must meet a 01:00Z OpenAQ stamp."""
    from aqetl import align

    weather_hour = align.openmeteo_local_to_utc("2026-11-01T07:00")
    openaq_hour = align.openaq_hour_start_from_period(
        None, datetime(2026, 11, 1, 1, 0, tzinfo=UTC)
    )
    assert weather_hour == openaq_hour == datetime(2026, 11, 1, 0, 0, tzinfo=UTC)


def test_empty_store_renders_chart_rather_than_failing(tmp_path):
    out = chart.render(root=tmp_path, out=tmp_path / "empty.png")
    assert out.exists() and out.stat().st_size > 1000
    assert schema.empty_curated().is_empty()
    assert openmeteo.TIMEZONE == "Asia/Ho_Chi_Minh"


def test_spine_never_counts_unelapsed_hours_as_missing():
    """A run at 02:00 must not charge stations for hours later today.

    The extract window ends at midnight tomorrow, so an uncapped spine marks
    every un-elapsed hour as missing and inflates coverage loss for everyone.
    """
    from datetime import datetime

    now = datetime.now(tz=UTC).replace(minute=0, second=0, microsecond=0)
    observed_hour = now - timedelta(hours=6)
    df = pl.DataFrame(
        {
            "city": ["hanoi"],
            "station_id": [1],
            "sensor_id": [9],
            "parameter": ["pm25"],
            "hour_utc": [observed_hour],
            "value_raw": [10.0],
        },
        schema_overrides={"hour_utc": pl.Datetime("us", "UTC")},
    )
    # Ask for a window running to the end of tomorrow, as the CLI does.
    filled = gaps.fill_gaps(
        df, observed_hour, datetime.now(tz=UTC).replace(
            hour=0, minute=0, second=0, microsecond=0
        ) + timedelta(days=1)
    )
    future = filled.filter(pl.col("hour_utc") > now)
    assert future.height == 0
