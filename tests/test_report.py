"""Rolling-window report.

The window is a RANGE over timestamps, so these pin elapsed-time behaviour and
the coverage filter rather than just the arithmetic.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, time, timedelta

import polars as pl
import pytest

from aqetl import report, schema, storage

TODAY = date(2026, 11, 20)


def _seed(root, values: dict[int, float], days: int = 12, parameter: str = "pm25"):
    """One station, one reading per hour, `values` mapping hour offset -> value."""
    start = datetime.combine(TODAY - timedelta(days=days - 1), datetime.min.time(), tzinfo=UTC)
    rows = []
    for hour in range(days * 24):
        value = values.get(hour, 10.0)
        rows.append(
            {
                "city": "hanoi",
                "station_id": 1,
                "station_name": "S",
                "sensor_id": 9,
                "parameter": parameter,
                "hour_utc": start + timedelta(hours=hour),
                "value_raw": value,
                "unit": "µg/m³",
                "unit_factor": 1.0,
                "value_ugm3": value,
                "coverage_pct": 100.0,
                "qc_negative": False,
                "qc_impossible": False,
                "qc_stuck": False,
                "qc_spike": False,
                "qc_missing": False,
                "qc_flags": 0,
                "value_clean": value,
            }
        )
    storage.write_curated(pl.DataFrame(rows, schema=schema.CURATED_SCHEMA), root=root)


def test_rolling_mean_grades_against_annual_guidelines(tmp_path):
    """90 ug/m3 sustained clears the 25 and 5 annual limits, not the 24 h one."""
    _seed(tmp_path, dict.fromkeys(range(12 * 24), 90.0))
    frame = report.build_rolling(tmp_path, days=12, today=TODAY)

    assert not frame.is_empty()
    # Once the window is full the mean is exactly the seeded level.
    last = frame.sort("day").tail(1).row(0, named=True)
    assert last["rolling7_ugm3"] == pytest.approx(90.0, abs=0.01)
    assert last["over_qcvn_annual"] is True
    assert last["over_who_annual"] is True
    assert last["qcvn_annual"] == 25.0
    assert last["who_annual"] == 5.0


def test_windows_without_enough_coverage_are_dropped(tmp_path):
    """Early days cannot have 96 of 168 hours, so they never appear."""
    _seed(tmp_path, dict.fromkeys(range(12 * 24), 30.0))
    frame = report.build_rolling(tmp_path, days=12, today=TODAY)

    # The first three days of a 12-day feed cannot fill a 7-day window.
    first_day = frame["day"].min()
    assert first_day >= TODAY - timedelta(days=8)
    assert frame["n7"].min() >= 96


def test_clean_air_is_below_the_annual_limits(tmp_path):
    _seed(tmp_path, dict.fromkeys(range(12 * 24), 2.0))
    frame = report.build_rolling(tmp_path, days=12, today=TODAY)

    assert frame["over_qcvn_annual"].any() is False
    assert frame["over_who_annual"].any() is False


def test_non_pm25_parameters_are_ignored_by_default(tmp_path):
    _seed(tmp_path, dict.fromkeys(range(12 * 24), 90.0), parameter="pm10")
    frame = report.build_rolling(tmp_path, days=12, today=TODAY)
    assert frame.is_empty()


def test_empty_store_returns_typed_empty_frame(tmp_path):
    frame = report.build_rolling(tmp_path, days=12, today=TODAY)
    assert frame.is_empty()
    assert dict(frame.schema) == report.ROLLING_SCHEMA


def test_readme_results_block_is_rewritten_between_markers(tmp_path):
    """README numbers must track the data, or they go stale on every run."""
    _seed(tmp_path, dict.fromkeys(range(12 * 24), 90.0))
    readme = tmp_path / "README.md"
    readme.write_text(
        "# T\n\n<!-- RESULTS:START -->\nstale 99 / 99\n<!-- RESULTS:END -->\n\nfooter\n"
    )

    assert report.update_readme(tmp_path, readme, today=TODAY) is True
    text = readme.read_text()

    assert "stale 99 / 99" not in text
    assert "Hanoi" in text
    assert text.startswith("# T")
    assert "footer" in text
    # Idempotent: a second pass with unchanged data writes nothing.
    assert report.update_readme(tmp_path, readme, today=TODAY) is False


def test_readme_without_markers_is_left_alone(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text("# No markers here\n")
    assert report.update_readme(tmp_path, readme, today=TODAY) is False
    assert readme.read_text() == "# No markers here\n"


def test_rolling_window_is_seven_days_not_six(tmp_path):
    """REGRESSION: INTERVAL 6 DAYS spans 145 hours, not the 168 of a 7-day window.

    The shipped store showed max n7 = 145, so the "7-day" mean was a 6-day mean
    and the 96-hour coverage gate silently meant 66% instead of 57%.
    """
    _seed(tmp_path, dict.fromkeys(range(12 * 24), 40.0))
    frame = report.build_rolling(tmp_path, days=12, today=TODAY)
    assert frame["n7"].max() == 168


def test_daily_window_must_end_on_the_day_it_labels(tmp_path):
    """REGRESSION: a window ending at 13:00 was published as that date's mean.

    17 of 393 station-days were attributed to the wrong date, because the chosen
    block was a trailing 24 h spanning mostly the previous day. A day that ends
    early must lose its count entirely, not be relabelled with stale data.
    """
    _seed(tmp_path, dict.fromkeys(range(12 * 24), 80.0))
    full = report.build_exceedance(tmp_path, days=12, today=TODAY).row(0, named=True)
    assert full["days_evaluated"] == 12

    # Truncate the final day at 13:00 and re-run.
    df = pl.read_parquet(tmp_path / "curated" / "city=hanoi" / "part-2026-11.parquet")
    cutoff = datetime.combine(TODAY, time(13, 0), tzinfo=UTC)
    kept = df.filter(pl.col("hour_utc") <= cutoff)
    for path in (tmp_path / "curated").rglob("*.parquet"):
        path.unlink()
    storage.write_curated(kept, root=tmp_path)

    truncated = report.build_exceedance(tmp_path, days=12, today=TODAY).row(0, named=True)
    assert truncated["days_evaluated"] == 11


def test_parameter_without_a_24h_limit_does_not_crash_the_readme(tmp_path):
    """REGRESSION: mean_ratio_qcvn is null for O3/SO2 and formatting it raised.

    That exception ran AFTER write_curated, so one O3 station reporting would
    fail every scheduled run and discard the data it had already fetched.
    """
    rows = []
    start = datetime.combine(TODAY - timedelta(days=9), datetime.min.time(), tzinfo=UTC)
    for hour in range(10 * 24):
        rows.append(
            {
                "city": "hanoi", "station_id": 1, "station_name": "S", "sensor_id": 9,
                "parameter": "o3", "hour_utc": start + timedelta(hours=hour),
                "value_raw": 50.0, "unit": "µg/m³", "unit_factor": 1.0, "value_ugm3": 50.0,
                "coverage_pct": 100.0, "qc_negative": False, "qc_impossible": False,
                "qc_stuck": False, "qc_spike": False, "qc_missing": False, "qc_flags": 0,
                "value_clean": 50.0,
            }
        )
    storage.write_curated(pl.DataFrame(rows, schema=schema.CURATED_SCHEMA), root=tmp_path)

    readme = tmp_path / "README.md"
    readme.write_text("<!-- RESULTS:START -->\n<!-- RESULTS:END -->\n")
    assert report.update_readme(tmp_path, readme, today=TODAY) is True
    assert "not regulated" in readme.read_text()
