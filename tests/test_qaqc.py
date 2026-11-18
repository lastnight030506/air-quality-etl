"""QC rules: each assertion pins a threshold boundary, not just the happy path."""

from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl
import pytest

from aqetl import qaqc

BASE = datetime(2026, 11, 1)


def frame(values: list[float | None], parameter: str = "pm25") -> pl.DataFrame:
    return pl.DataFrame(
        {
            "city": ["hanoi"] * len(values),
            "station_id": [1] * len(values),
            "sensor_id": [9] * len(values),
            "parameter": [parameter] * len(values),
            "hour_utc": [BASE + timedelta(hours=i) for i in range(len(values))],
            "value_ugm3": values,
        },
        schema_overrides={"hour_utc": pl.Datetime("us", "UTC")},
    )


def test_stuck_fires_at_exactly_six_identical_values():
    out = qaqc.apply_all(frame([10.0] * 5 + [20.0] * 6 + [30.0]))
    assert out["qc_stuck"].sum() == 6
    assert out.filter(pl.col("value_ugm3") == 20.0)["qc_stuck"].all()


def test_stuck_does_not_fire_at_five():
    out = qaqc.apply_all(frame([10.0] * 5 + [30.0]))
    assert out["qc_stuck"].sum() == 0


def test_stuck_ignores_sub_cent_noise():
    """A sensor wobbling in the third decimal is noisy, not stuck."""
    values = [10.0, 10.001, 10.002, 10.003, 10.004, 10.005, 10.006]
    out = qaqc.apply_all(frame(values))
    assert out["qc_stuck"].sum() == 0


def test_negative_is_flagged_and_nulled():
    out = qaqc.apply_all(frame([10.0, -3.0, 10.0]))
    flagged = out.filter(pl.col("value_ugm3") == -3.0)
    assert flagged["qc_negative"].all()
    assert flagged["value_clean"].is_null().all()


def test_impossible_above_ceiling_is_flagged_and_nulled():
    out = qaqc.apply_all(frame([10.0, 5000.0, 10.0]))
    row = out.filter(pl.col("value_ugm3") == 5000.0)
    assert row["qc_impossible"].all()
    assert row["value_clean"].is_null().all()


def test_ceiling_is_per_parameter():
    """2000 ug/m3 is impossible for O3 but merely extreme for PM10."""
    assert qaqc.apply_all(frame([1500.0], parameter="o3"))["qc_impossible"].all()
    assert not qaqc.apply_all(frame([1500.0], parameter="pm10"))["qc_impossible"].any()


def test_spike_kept_not_dropped():
    """A dust event is the thing we exist to chart; flag it, never erase it."""
    out = qaqc.apply_all(frame([10.0] * 5 + [500.0] + [10.0] * 5))
    row = out.filter(pl.col("value_ugm3") == 500.0)
    assert row["qc_spike"].all()
    assert row["value_clean"].to_list() == [500.0]


def test_spike_needs_absolute_floor():
    """A quiet sensor must not flag every ordinary hour."""
    out = qaqc.apply_all(frame([3.0] * 8 + [20.0] + [3.0] * 8))
    assert out["qc_spike"].sum() == 0


def test_flag_count_sums_flags():
    out = qaqc.apply_all(frame([10.0, -3.0, 6000.0]))
    counts = out["qc_flags"].to_list()
    assert counts[0] == 0
    assert counts[1] == 1
    assert counts[2] >= 1


def test_empty_frame_survives():
    empty = qaqc.apply_all(pl.DataFrame({"value_ugm3": []}, schema={"value_ugm3": pl.Float64()}))
    assert empty.is_empty()


@pytest.mark.parametrize("parameter", ["pm25", "pm10", "no2", "o3"])
def test_null_values_do_not_crash_rules(parameter):
    out = qaqc.apply_all(frame([10.0, None, 12.0], parameter=parameter))
    assert len(out) == 3


def test_stuck_never_fires_on_gap_filled_nulls():
    """A stretch of absent hours is not a frozen sensor.

    Regression: nulls compare equal, so run-length grouping marked every
    gap-filled hour as stuck and inflated the anomaly rate to ~50%.
    """
    out = qaqc.apply_all(frame([10.0, 11.0] + [None] * 10 + [12.0, 13.0]))
    assert out["qc_missing"].sum() == 10
    assert out["qc_stuck"].sum() == 0


def test_stuck_still_fires_on_a_real_run_next_to_nulls():
    out = qaqc.apply_all(frame([None] * 8 + [40.0] * 6 + [None] * 8))
    stuck = out.filter(pl.col("qc_stuck"))
    assert stuck.height == 6
    assert stuck["value_ugm3"].to_list() == [40.0] * 6
