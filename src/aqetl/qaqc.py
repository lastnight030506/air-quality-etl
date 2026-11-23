"""QA/QC rules.

Each rule is a pure Polars expression that produces a boolean column. The rules
never drop rows: a flagged row stays in the store with its flag set, because the
anomaly rate is itself a deliverable and Excel-style silent deletion would hide
it. `value_clean` is nulled where a rule invalidates the reading.

Flag semantics
-------------
qc_negative     value < 0                      physically impossible
qc_impossible   above a hard sensor ceiling    telemetry corruption
qc_stuck        >= STUCK_RUN identical values  frozen/failed sensor
qc_spike        > SPIKE_FACTOR * 24h median AND > SPIKE_FLOOR
qc_missing      hour absent from the spine     outage, not a reading

A physically impossible value is both `qc_negative`/`qc_impossible` and nulled.
A spike is flagged but KEPT: a real dust event looks like a spike, and throwing
it away would erase the most interesting pollution this project exists to chart.
"""

from __future__ import annotations

import polars as pl

from .schema import QC_FLAG_COLUMNS

STUCK_RUN = 6
SPIKE_FACTOR = 5.0
SPIKE_FLOOR = 100.0

# Upper bounds for a single hourly mean. Set well above anything Vietnam
# records (PM2.5 peaks around 150-200) so only corrupt data trips them.
IMPOSSIBLE_CEILING: dict[str, float] = {
    "pm25": 1000.0,
    "pm10": 2000.0,
    "no2": 2000.0,
    "o3": 1000.0,
}

# Physical lower bound is zero for concentrations; the negative rule covers it.

_SENSOR = ["city", "station_id", "sensor_id"]


def flag_negative(df: pl.DataFrame, col: str = "value_ugm3") -> pl.DataFrame:
    return df.with_columns((pl.col(col) < 0).fill_null(False).alias("qc_negative"))


def flag_impossible(df: pl.DataFrame, col: str = "value_ugm3") -> pl.DataFrame:
    ceiling = pl.col("parameter").replace_strict(
        IMPOSSIBLE_CEILING, default=None, return_dtype=pl.Float64
    )
    return df.with_columns(
        (pl.col(col) > ceiling).fill_null(False).alias("qc_impossible")
    )


def flag_stuck(df: pl.DataFrame, col: str = "value_ugm3") -> pl.DataFrame:
    """True for every row in a run of >= STUCK_RUN identical consecutive values.

    Runs are computed per sensor over `hour_utc` order, on the raw value rounded
    to 2 dp: sensor noise at the 3rd decimal is not a stuck sensor.

    Nulls are excluded explicitly. SQL null comparison is not equality, but a
    run-length grouping treats every null as the same value, so a stretch of
    gap-filled hours otherwise reads as a frozen sensor and inflates the anomaly
    rate by an order of magnitude.
    """
    ordered = df.sort([*_SENSOR, "hour_utc"])
    runs = (
        ordered
        # An all-null column arrives as dtype Null, which round() rejects; cast
        # first so a fully empty parameter cannot crash the run.
        .with_columns(pl.col(col).cast(pl.Float64).round(2).alias("_v"))
        .with_columns(
            pl.struct([*_SENSOR, "_v"]).rle_id().over(_SENSOR).alias("_run")
        )
        .with_columns(pl.len().over([*_SENSOR, "_run"]).alias("_run_len"))
        .with_columns(
            (
                (pl.col("_run_len") >= STUCK_RUN) & pl.col("_v").is_not_null()
            )
            .fill_null(False)
            .alias("qc_stuck")
        )
        .drop("_v", "_run", "_run_len")
    )
    return runs.sort(["city", "station_id", "parameter", "hour_utc"])


def flag_spike(df: pl.DataFrame, col: str = "value_ugm3") -> pl.DataFrame:
    """Needs both a relative and an absolute condition.

    The absolute floor stops a quiet sensor (median 3 ug/m3) from having every
    ordinary 20 ug/m3 hour labelled a spike. The rule looks at +/-12h so it does
    not compare a value against a window that already contains it.
    """
    ordered = df.sort([*_SENSOR, "hour_utc"])
    rolled = ordered.with_columns(
        pl.col(col)
        .rolling_median(window_size=25, center=True, min_samples=6)
        .over(_SENSOR)
        .alias("_median24")
    )
    return (
        rolled.with_columns(
            (
                (pl.col(col) > SPIKE_FACTOR * pl.col("_median24"))
                & (pl.col(col) > SPIKE_FLOOR)
            )
            .fill_null(False)
            .alias("qc_spike")
        )
        .drop("_median24")
        .sort(["city", "station_id", "parameter", "hour_utc"])
    )


def flag_missing(df: pl.DataFrame, col: str = "value_ugm3") -> pl.DataFrame:
    """Flag rows the spine invented, not rows we received but could not convert.

    `gaps.fill_gaps` already set qc_missing from the spine, and that is the
    authoritative signal: a gap-filled row has no coverage figure because the
    station never reported it. A received row with an unrecognised unit also has
    a null value_ugm3, but it still carries coverage — counting it as missing
    would blame the station's uptime for our conversion gap.
    """
    if "coverage_pct" not in df.columns:
        # Frames that never went through the spine carry no coverage column; the
        # null test is then the only signal available.
        return df.with_columns(pl.col(col).is_null().alias("qc_missing"))
    return df.with_columns(
        (pl.col(col).is_null() & pl.col("coverage_pct").is_null()).alias("qc_missing")
    )


def add_flag_count(df: pl.DataFrame) -> pl.DataFrame:
    bits = [pl.col(c).fill_null(False).cast(pl.Int32) for c in QC_FLAG_COLUMNS]
    return df.with_columns(pl.sum_horizontal(bits).alias("qc_flags"))


def value_clean(df: pl.DataFrame, col: str = "value_ugm3") -> pl.DataFrame:
    """Null out readings a rule has invalidated.

    Spike is deliberately excluded: see module docstring.
    """
    invalid = (
        pl.col("qc_negative").fill_null(False)
        | pl.col("qc_impossible").fill_null(False)
        | pl.col("qc_stuck").fill_null(False)
        | pl.col("qc_missing").fill_null(False)
    )
    return df.with_columns(
        pl.when(invalid).then(None).otherwise(pl.col(col)).alias("value_clean")
    )


def apply_all(df: pl.DataFrame) -> pl.DataFrame:
    """Run every rule in the order the docstring describes."""
    if df.is_empty():
        return df
    out = df
    out = flag_negative(out)
    out = flag_impossible(out)
    out = flag_stuck(out)
    out = flag_spike(out)
    out = flag_missing(out)
    out = add_flag_count(out)
    out = value_clean(out)
    return out
