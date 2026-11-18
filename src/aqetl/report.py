"""Reporting queries.

All aggregation happens in DuckDB over the Parquet store; this module only
supplies the date window and the benchmark limits, so a limit change never means
rewriting SQL. Every frame is rebuilt from disk, which is what lets `aqetl report`
refresh the README without touching the API.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path

import duckdb
import polars as pl

from . import standards, storage
from .config import PARAMETER_ORDER

DEFAULT_ROOT = Path("data")

_DAILY_SQL = """
SELECT CAST(hour_utc AS DATE) AS date,
       city,
       parameter,
       avg(value_clean) AS mean_ugm3,
       -- Distinct hours, not rows: several stations per city would otherwise
       -- report 58 "hours" in a 24-hour day and pass any coverage test.
       count(DISTINCT hour_utc) FILTER (WHERE value_clean IS NOT NULL) AS n_hours
FROM curated
WHERE hour_utc >= ? AND hour_utc < ?
GROUP BY 1, 2, 3
ORDER BY date, city, parameter
"""

# Anomaly rate is split into two numbers on purpose. A station that is offline
# has no readings to be suspicious of; calling that a 100% anomaly rate makes
# the whole metric useless, because coverage and data quality are different
# failures with different fixes. `anomaly_pct` is therefore bad readings as a
# share of hours actually RECEIVED, and `missing_pct` carries coverage.
_STATION_QC_SQL = """
SELECT city,
       station_id,
       coalesce(nullif(trim(min(station_name)), ''), 'station ' || cast(station_id AS VARCHAR))
           AS station_name,
       parameter,
       count(*) AS n_expected,
       count(*) FILTER (WHERE coalesce(qc_missing, false)) AS qc_missing,
       count(*) FILTER (WHERE NOT coalesce(qc_missing, false)) AS n_received,
       count(*) FILTER (WHERE coalesce(qc_negative, false)) AS qc_negative,
       count(*) FILTER (WHERE coalesce(qc_impossible, false)) AS qc_impossible,
       count(*) FILTER (WHERE coalesce(qc_stuck, false)) AS qc_stuck,
       count(*) FILTER (WHERE coalesce(qc_spike, false)) AS qc_spike,
       100.0 * count(*) FILTER (WHERE coalesce(qc_missing, false)) / count(*) AS missing_pct,
       100.0 * count(*) FILTER (
           WHERE NOT coalesce(qc_missing, false) AND coalesce(qc_flags, 0) > 0
       ) / nullif(count(*) FILTER (WHERE NOT coalesce(qc_missing, false)), 0) AS anomaly_pct
FROM curated
WHERE hour_utc >= ? AND hour_utc < ?
GROUP BY city, station_id, parameter
ORDER BY city, station_id, parameter
"""

# 24h mean needs 24 hourly values spanning exactly 23 h; a day contributes its
# last qualifying hour, so a station-day is counted once.
_EXCEEDANCE_SQL = """
WITH hourly AS (
    SELECT city, station_id, parameter, hour_utc, value_clean
    FROM curated
    WHERE hour_utc >= ? AND hour_utc < ?
), rolled AS (
    -- The store carries a gap-filled row per expected hour, so a 24-row window
    -- is 24 clock hours and the coverage test is on non-null clean values.
    SELECT city, station_id, parameter, hour_utc,
           CAST(hour_utc AS DATE) AS d,
           avg(value_clean) OVER w AS m24,
           count(value_clean) OVER w AS n24,
           date_diff('hour', min(hour_utc) OVER w, hour_utc) AS span_h
    FROM hourly
    WINDOW w AS (PARTITION BY city, station_id, parameter ORDER BY hour_utc
                 ROWS BETWEEN 23 PRECEDING AND CURRENT ROW)
), full_days AS (
    SELECT * FROM rolled WHERE span_h = 23 AND n24 >= 18
), day_last AS (
    SELECT *, row_number() OVER (PARTITION BY city, station_id, parameter, d
                                ORDER BY hour_utc DESC) AS rn
    FROM full_days
), station_day AS (
    SELECT t.city, t.parameter, t.d, t.m24, lmt.qcvn, lmt.who
    FROM day_last t
    JOIN limits lmt ON lmt.parameter = t.parameter AND lmt.d = t.d
    WHERE t.rn = 1
)
SELECT city,
       parameter,
       CAST(count(DISTINCT d) AS INTEGER) AS days_evaluated,
       CAST(count(DISTINCT d) FILTER (WHERE m24 > qcvn) AS INTEGER) AS days_over_qcvn,
       CAST(count(DISTINCT d) FILTER (WHERE m24 > who) AS INTEGER) AS days_over_who,
       avg(CASE WHEN qcvn > 0 THEN m24 / qcvn END) AS mean_ratio_qcvn
FROM station_day
GROUP BY city, parameter
ORDER BY city, parameter
"""


def _window(days: int, today: date) -> tuple[datetime, datetime]:
    """Half-open [start, end) over the trailing `days` calendar days, UTC."""
    start = datetime.combine(today - timedelta(days=days - 1), time.min, tzinfo=UTC)
    return start, datetime.combine(today + timedelta(days=1), time.min, tzinfo=UTC)


# Declared outputs: DuckDB on an empty store yields Object columns, which parquet
# cannot take, and the report is written on a fresh checkout too.
DAILY_SCHEMA = {
    "date": pl.Date,
    "city": pl.String,
    "parameter": pl.String,
    "mean_ugm3": pl.Float64,
    "n_hours": pl.Int64,
}

STATION_QC_SCHEMA = {
    "city": pl.String,
    "station_id": pl.Int64,
    "station_name": pl.String,
    "parameter": pl.String,
    "n_expected": pl.Int64,
    "qc_missing": pl.Int64,
    "n_received": pl.Int64,
    "qc_negative": pl.Int64,
    "qc_impossible": pl.Int64,
    "qc_stuck": pl.Int64,
    "qc_spike": pl.Int64,
    "missing_pct": pl.Float64,
    "anomaly_pct": pl.Float64,
}

EXCEEDANCE_SCHEMA = {
    "city": pl.String,
    "parameter": pl.String,
    "days_evaluated": pl.Int32,
    "days_over_qcvn": pl.Int32,
    "days_over_who": pl.Int32,
    "mean_ratio_qcvn": pl.Float64,
}


def _conform(frame: pl.DataFrame, contract: dict[str, pl.DataType]) -> pl.DataFrame:
    return frame.select(pl.col(name).cast(dtype) for name, dtype in contract.items())


def _with_limits(con: duckdb.DuckDBPyConnection, days: int, today: date) -> None:
    """Load (parameter, day, limit) for the window; limits resolved in standards."""
    con.execute(
        "CREATE TABLE limits (parameter VARCHAR, d DATE, qcvn DOUBLE, who DOUBLE)"
    )
    rows = [
        (
            parameter,
            day,
            standards.limit_for("qcvn", parameter, "24h", day),
            standards.limit_for("who", parameter, "24h", day),
        )
        for day in pl.date_range(
            today - timedelta(days=days - 1), today, interval="1d", eager=True
        ).to_list()
        for parameter in PARAMETER_ORDER
    ]
    con.executemany("INSERT INTO limits VALUES (?, ?, ?, ?)", rows)


def build_daily(
    root: Path = DEFAULT_ROOT, days: int = 30, today: date | None = None
) -> pl.DataFrame:
    start, end = _window(days, today or date.today())
    con = storage.duck_conn(root)
    try:
        return _conform(storage.fetch(con, _DAILY_SQL, [start, end]), DAILY_SCHEMA)
    finally:
        con.close()


def build_station_qc(
    root: Path = DEFAULT_ROOT, days: int = 30, today: date | None = None
) -> pl.DataFrame:
    start, end = _window(days, today or date.today())
    con = storage.duck_conn(root)
    try:
        return _conform(
            storage.fetch(con, _STATION_QC_SQL, [start, end]), STATION_QC_SCHEMA
        )
    finally:
        con.close()


def build_exceedance(
    root: Path = DEFAULT_ROOT, days: int = 90, today: date | None = None
) -> pl.DataFrame:
    today = today or date.today()
    start, end = _window(days, today)
    con = storage.duck_conn(root)
    try:
        _with_limits(con, days, today)
        return _conform(
            storage.fetch(con, _EXCEEDANCE_SQL, [start, end]), EXCEEDANCE_SCHEMA
        )
    finally:
        con.close()


def write_report(
    root: Path = DEFAULT_ROOT, today: date | None = None
) -> dict[str, Path]:
    today = today or date.today()
    out = root / "reports"
    out.mkdir(parents=True, exist_ok=True)
    frames = {
        "daily": build_daily(root, 30, today),
        "station_qc": build_station_qc(root, 30, today),
        "exceedance": build_exceedance(root, 90, today),
    }
    paths: dict[str, Path] = {}
    for name, frame in frames.items():
        paths[name] = out / f"{name}.parquet"
        frame.write_parquet(paths[name], compression="zstd")
    return paths


__all__ = [
    "build_daily",
    "build_exceedance",
    "build_station_qc",
    "write_report",
]
