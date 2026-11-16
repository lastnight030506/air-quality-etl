"""The one schema every stage agrees on.

Curated rows are keyed by (station_id, parameter, hour_utc). `hour_utc` is the
START of the hour. OpenAQ stamps hours with the END of the period, so the
extract layer shifts by -1 h; see align.py.
"""

from __future__ import annotations

import polars as pl

CURATED_KEY: tuple[str, ...] = ("city", "station_id", "parameter", "hour_utc")

CURATED_SCHEMA: dict[str, pl.DataType] = {
    "city": pl.String(),
    "station_id": pl.Int64(),
    "station_name": pl.String(),
    "sensor_id": pl.Int64(),
    "parameter": pl.String(),
    "hour_utc": pl.Datetime("us", "UTC"),
    "value_raw": pl.Float64(),
    "unit": pl.String(),
    "unit_factor": pl.Float64(),
    "value_ugm3": pl.Float64(),
    "coverage_pct": pl.Float64(),
    "qc_negative": pl.Boolean(),
    "qc_impossible": pl.Boolean(),
    "qc_stuck": pl.Boolean(),
    "qc_spike": pl.Boolean(),
    "qc_missing": pl.Boolean(),
    "qc_flags": pl.Int32(),
    "value_clean": pl.Float64(),
}

WEATHER_SCHEMA: dict[str, pl.DataType] = {
    "city": pl.String(),
    "hour_utc": pl.Datetime("us", "UTC"),
    "temperature_2m": pl.Float64(),
    "relative_humidity_2m": pl.Float64(),
    "wind_speed_10m": pl.Float64(),
    "precipitation": pl.Float64(),
}

QC_FLAG_COLUMNS: tuple[str, ...] = (
    "qc_negative",
    "qc_impossible",
    "qc_stuck",
    "qc_spike",
    "qc_missing",
)


def empty_curated() -> pl.DataFrame:
    return pl.DataFrame(schema=CURATED_SCHEMA)


def empty_weather() -> pl.DataFrame:
    return pl.DataFrame(schema=WEATHER_SCHEMA)


def conform(df: pl.DataFrame) -> pl.DataFrame:
    """Cast to the contract. Raises if a required column is missing."""
    missing = set(CURATED_SCHEMA) - set(df.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    return df.select(
        [pl.col(name).cast(dtype) for name, dtype in CURATED_SCHEMA.items()]
    ).sort(list(CURATED_KEY))
