"""Timestamp alignment.

The failure this file guards against is silent: a one-hour shift produces no
error, just every number slightly wrong. If someone later "cleans up" the -1h
in openaq_hour_start, these tests must go red.
"""

from __future__ import annotations

from datetime import UTC, datetime

from aqetl import align


def test_openaq_stamp_is_time_ending_so_shift_back_one_hour():
    """OpenAQ 03:00Z covers 02:00-02:59, so the hour STARTS at 02:00Z."""
    stamp = datetime(2026, 11, 1, 3, 0, tzinfo=UTC)
    assert align.openaq_hour_start(stamp) == datetime(2026, 11, 1, 2, 0, tzinfo=UTC)


def test_explicit_period_start_wins_over_the_shift():
    """When the payload says where the period began, believe it."""
    start = datetime(2026, 11, 1, 2, 0, tzinfo=UTC)
    end = datetime(2026, 11, 1, 3, 0, tzinfo=UTC)
    assert align.openaq_hour_start_from_period(start, end) == start


def test_absent_period_falls_back_to_shifting():
    end = datetime(2026, 11, 1, 3, 0, tzinfo=UTC)
    assert align.openaq_hour_start_from_period(None, end) == datetime(
        2026, 11, 1, 2, 0, tzinfo=UTC
    )


def test_openmeteo_local_becomes_utc_start_of_same_hour():
    """Open-Meteo '03:00' is local ICT (UTC+7) and starts that hour -> 20:00Z prior day."""
    assert align.openmeteo_local_to_utc("2026-11-01T03:00") == datetime(
        2026, 10, 31, 20, 0, tzinfo=UTC
    )


def test_openmeteo_and_openaq_land_in_the_same_bucket():
    """The whole point: one local 03:00 reading and one OpenAQ hour agree.

    03:00 ICT on Nov 1 is 20:00Z on Oct 31, and the hour starting there ends at
    21:00Z, which is the stamp OpenAQ publishes.
    """
    weather_hour = align.openmeteo_local_to_utc("2026-11-01T03:00")
    openaq_hour = align.openaq_hour_start_from_period(
        None, datetime(2026, 10, 31, 21, 0, tzinfo=UTC)
    )
    assert weather_hour == openaq_hour


def test_spine_covers_every_hour_inclusive():
    spine = align.hourly_spine(
        datetime(2026, 11, 1, 0, 0, tzinfo=UTC),
        datetime(2026, 11, 1, 3, 0, tzinfo=UTC),
        city="hanoi",
        station_id=1,
        parameter="pm25",
    )
    assert len(spine) == 4
    assert spine["city"].unique().to_list() == ["hanoi"]


def test_join_weather_on_city_and_hour():
    import polars as pl

    curated = pl.DataFrame(
        {
            "city": ["hanoi", "hanoi"],
            "hour_utc": [
                datetime(2026, 11, 1, 0, 0, tzinfo=UTC),
                datetime(2026, 11, 1, 1, 0, tzinfo=UTC),
            ],
            "parameter": ["pm25", "pm25"],
        },
        schema_overrides={"hour_utc": pl.Datetime("us", "UTC")},
    )
    weather = pl.DataFrame(
        {
            "city": ["hanoi"],
            "hour_utc": [datetime(2026, 11, 1, 0, 0, tzinfo=UTC)],
            "temperature_2m": [27.5],
            "relative_humidity_2m": [80.0],
            "wind_speed_10m": [5.0],
            "precipitation": [0.0],
        },
        schema_overrides={"hour_utc": pl.Datetime("us", "UTC")},
    )
    joined = align.join_weather(curated, weather)
    assert joined["temperature_2m"].to_list() == [27.5, None]


def test_join_weather_overwrites_the_contract_columns():
    """REGRESSION: curated already carries the weather columns as nulls.

    Joining a second copy made polars suffix it `_right` and leave every reading
    with a null temperature, so the whole weather fetch was silently discarded.
    """
    import polars as pl

    from aqetl import schema

    curated = pl.DataFrame(
        {
            "city": ["hanoi"],
            "hour_utc": [datetime(2026, 11, 1, 0, 0, tzinfo=UTC)],
            "parameter": ["pm25"],
            "value_clean": [20.0],
            "temperature_2m": [None],
            "relative_humidity_2m": [None],
            "wind_speed_10m": [None],
            "precipitation": [None],
        },
        schema_overrides={
            "hour_utc": pl.Datetime("us", "UTC"),
            "temperature_2m": pl.Float64(),
            "relative_humidity_2m": pl.Float64(),
            "wind_speed_10m": pl.Float64(),
            "precipitation": pl.Float64(),
        },
    )
    weather = pl.DataFrame(
        {
            "city": ["hanoi"],
            "hour_utc": [datetime(2026, 11, 1, 0, 0, tzinfo=UTC)],
            "temperature_2m": [28.4],
            "relative_humidity_2m": [80.0],
            "wind_speed_10m": [5.0],
            "precipitation": [0.0],
        },
        schema_overrides={"hour_utc": pl.Datetime("us", "UTC")},
    )

    joined = align.join_weather(curated, weather)
    assert joined["temperature_2m"].to_list() == [28.4]
    assert not any(c.endswith("_right") for c in joined.columns)
    # Still conforms to the store contract.
    assert set(joined.columns) <= set(schema.CURATED_SCHEMA)
