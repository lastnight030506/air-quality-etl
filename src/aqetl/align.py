"""Timestamp alignment.

This module exists because the two sources disagree about what a timestamp
means, and getting it wrong shifts every joined row by an hour without any
error surfacing.

* OpenAQ uses an exclusive time-ENDING convention: a stamp of 03:00 covers
  02:00-02:59. The hourly payload usually carries an explicit `period` with
  `datetimeFrom`/`datetimeTo`; when it does we trust `datetimeFrom`, and when
  it does not we subtract one hour from the stamp.
* Open-Meteo returns local wall-clock strings that are time-STARTING: "03:00"
  means the hour beginning at 03:00 local.

Everything downstream keys on `hour_utc`, the START of the hour, in UTC.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import polars as pl

VN_TZ = ZoneInfo("Asia/Ho_Chi_Minh")


def openaq_hour_start(stamp_utc: datetime) -> datetime:
    """OpenAQ's time-ending stamp -> start of the hour it describes."""
    return stamp_utc - timedelta(hours=1)


def openaq_hour_start_from_period(from_utc: datetime | None, to_utc: datetime) -> datetime:
    """Prefer the explicit period start; fall back to shifting the end stamp."""
    if from_utc is not None:
        return from_utc
    return openaq_hour_start(to_utc)


def openmeteo_local_to_utc(local_stamp: str) -> datetime:
    """Open-Meteo local naive timestamp -> UTC, keeping the hour it starts."""
    naive = datetime.fromisoformat(local_stamp)
    if naive.tzinfo is None:
        naive = naive.replace(tzinfo=VN_TZ)
    return naive.astimezone(ZoneInfo("UTC"))


def align_frame(df: pl.DataFrame) -> pl.DataFrame:
    """Normalise any frame carrying a mixed-case stamp column to hour_utc.

    Kept as a function so tests can pin the behaviour and so a future source
    has exactly one place to plug into.
    """
    if "hour_utc" in df.columns:
        return df.with_columns(pl.col("hour_utc").dt.replace_time_zone("UTC"))
    raise ValueError("frame has no hour_utc column")


def join_weather(
    curated: pl.DataFrame, weather: pl.DataFrame
) -> pl.DataFrame:
    """Left-join hourly weather onto observations on (city, hour_utc)."""
    weather_cols = (
        "temperature_2m",
        "relative_humidity_2m",
        "wind_speed_10m",
        "precipitation",
    )
    if weather.is_empty():
        return curated.with_columns(
            *[pl.lit(None, dtype=pl.Float64).alias(name) for name in weather_cols]
        )
    # The curated contract already carries these columns as nulls, so they must be
    # dropped first: polars would otherwise suffix the joined copies `_right` and
    # leave every reading with a null weather value.
    return curated.drop(*weather_cols, strict=False).join(
        weather.select("city", "hour_utc", *weather_cols),
        on=["city", "hour_utc"],
        how="left",
    )


def hourly_spine(
    start: datetime, end: datetime, city: str, station_id: int, parameter: str
) -> pl.DataFrame:
    """Every hour that SHOULD exist, so gaps can be detected rather than ignored.

    OpenAQ returns only the hours it has, so a silent outage looks identical to
    a clean run unless the expected spine is rebuilt explicitly.
    """
    hours = pl.datetime_range(
        start, end, interval="1h", eager=True, time_zone="UTC"
    ).alias("hour_utc")
    n = len(hours)
    return pl.DataFrame(
        {
            "city": [city] * n,
            "station_id": [station_id] * n,
            "parameter": [parameter] * n,
            "hour_utc": hours,
        }
    )
