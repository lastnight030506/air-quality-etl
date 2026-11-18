"""Gap filling.

OpenAQ returns only the hours it has, so an outage and a clean run look the same
unless the expected spine is rebuilt. `fill_gaps` materialises every hour per
(city, station, parameter) and marks the absent ones `qc_missing`, so the
missing rate becomes a number instead of an assumption.

The curated key does not carry `sensor_id`, so when a station exposes two sensors
for one parameter the first sensor's rows win: one row per station-hour.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl

from . import align, schema
from .schema import CURATED_KEY, CURATED_SCHEMA, QC_FLAG_COLUMNS

_GROUP = ["city", "station_id", "parameter"]


def _ensure_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Extract produces a subset of the schema; add the rest as typed nulls."""
    missing = {
        name: pl.lit(None, dtype=dtype)
        for name, dtype in CURATED_SCHEMA.items()
        if name not in df.columns
    }
    return df.with_columns(**missing) if missing else df


def fill_gaps(df: pl.DataFrame, start: datetime, end: datetime) -> pl.DataFrame:
    """`end` is exclusive, matching the half-open window used by report._window.

    Treating it as inclusive would invent one absent hour at every window
    boundary and charge it to the station as a missing reading.
    """
    if df.is_empty():
        return schema.empty_curated()
    last_hour = end - timedelta(hours=1)
    if last_hour < start:
        return schema.empty_curated()
    ordered = _ensure_columns(df).sort(list(CURATED_KEY))
    frames = []
    for (city, station_id, parameter), group in ordered.group_by(
        _GROUP, maintain_order=True
    ):
        # A station's expected hours start at its own first reading inside the
        # window: charging it for the months before it appeared buries the real
        # gaps in a 95% figure. The END stays at the window edge on purpose —
        # a station that goes dark mid-window is an outage worth flagging.
        received = group.filter(pl.col("value_raw").is_not_null())
        if received.is_empty():
            continue
        spine_start = max(start, received["hour_utc"].min())
        spine_end = end - timedelta(hours=1)
        spine = align.hourly_spine(spine_start, spine_end, city, station_id, parameter)
        # The curated key has no sensor_id, so two sensors on one parameter would
        # collide: prefer a real reading over a null, then the lowest sensor id.
        observed = (
            group.sort(pl.col("value_raw").is_null(), "sensor_id")
            .unique(subset=["hour_utc"], keep="first")
            .drop(*_GROUP)
            .with_columns(pl.lit(True).alias("_present"))
        )
        filled = spine.join(observed, on="hour_utc", how="left").with_columns(
            pl.col("_present").fill_null(False)
        )
        present = pl.col("_present")
        flags = [c for c in QC_FLAG_COLUMNS if c != "qc_missing"]
        frames.append(
            filled.with_columns(
                *[pl.col(c).fill_null(False) for c in flags],
                # `qc_missing` is assigned, never inherited: the spine decides.
                (~present).alias("qc_missing"),
            ).with_columns(
                # Filled rows carry no flags; observed rows keep whatever the
                # QC stage already counted.
                pl.when(present)
                .then(pl.col("qc_flags").fill_null(0))
                .otherwise(0)
                .cast(pl.Int32)
                .alias("qc_flags"),
            ).drop("_present")
        )
    out = pl.concat(frames, how="vertical")
    return out.select(
        [pl.col(name).cast(dtype) for name, dtype in CURATED_SCHEMA.items()]
    ).sort(list(CURATED_KEY))


def missing_rate(df: pl.DataFrame) -> pl.DataFrame:
    expected = pl.len().cast(pl.Int64).alias("expected_hours")
    missing = pl.col("qc_missing").sum().cast(pl.Int64).alias("missing_hours")
    return (
        df.group_by(_GROUP)
        .agg(expected, missing)
        .with_columns(
            (100.0 * pl.col("missing_hours") / pl.col("expected_hours")).alias(
                "missing_pct"
            )
        )
        .select(*_GROUP, "expected_hours", "missing_hours", "missing_pct")
        .sort(_GROUP)
    )
