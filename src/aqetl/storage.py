"""Parquet store.

Monthly part files are the unit of write, because that is the granularity the
retention prune can drop whole and a re-fetched month is only ever rewritten,
not appended to. A write is an upsert on the curated key so re-running a window
replaces the hours it covers instead of duplicating them.
"""

from __future__ import annotations

import calendar
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import duckdb
import polars as pl

from . import schema
from .config import RETENTION_DAYS
from .schema import CURATED_KEY, WEATHER_SCHEMA

DEFAULT_ROOT = Path("data")


def _month_end(month: str) -> date:
    year, mon = (int(part) for part in month.split("-"))
    return date(year, mon, calendar.monthrange(year, mon)[1])


def _upsert(path: Path, new: pl.DataFrame, key: list[str]) -> None:
    """Concat with whatever the file already holds, newest row wins per key."""
    if path.exists():
        old = pl.read_parquet(path)
        frame = pl.concat([old.select(new.columns), new], how="vertical")
    else:
        frame = new
    frame.unique(subset=key, keep="last").sort(key).write_parquet(
        path, compression="zstd"
    )


def _write_parts(
    df: pl.DataFrame, root: Path, kind: str, key: list[str]
) -> list[Path]:
    if df.is_empty():
        return []
    stamped = df.with_columns(pl.col("hour_utc").dt.strftime("%Y-%m").alias("_part"))
    written: list[Path] = []
    for (city, month), part in stamped.partition_by(
        ["city", "_part"], as_dict=True, maintain_order=True
    ).items():
        path = root / kind / f"city={city}" / f"part-{month}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        _upsert(path, part.drop("_part"), key)
        written.append(path)
    return sorted(written)


def write_curated(df: pl.DataFrame, root: Path = DEFAULT_ROOT) -> list[Path]:
    return _write_parts(schema.conform(df), root, "curated", list(CURATED_KEY))


def write_weather(df: pl.DataFrame, root: Path = DEFAULT_ROOT) -> list[Path]:
    missing = set(WEATHER_SCHEMA) - set(df.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    frame = df.select(
        [pl.col(name).cast(dtype) for name, dtype in WEATHER_SCHEMA.items()]
    )
    return _write_parts(frame, root, "weather", ["city", "hour_utc"])


def fetch(con: duckdb.DuckDBPyConnection, sql: str, params: list | None = None) -> pl.DataFrame:
    """Run SQL and return polars. DuckDB 2.0 has no polars bridge without pyarrow."""
    return pl.DataFrame(con.execute(sql, params or []).fetchnumpy())


_SQL_TYPES = {
    pl.String: "VARCHAR",
    pl.Int64: "BIGINT",
    pl.Int32: "INTEGER",
    pl.Float64: "DOUBLE",
    pl.Boolean: "BOOLEAN",
    pl.Datetime("us", "UTC"): "TIMESTAMP WITH TIME ZONE",
    pl.Datetime("us"): "TIMESTAMP",
    pl.Date: "DATE",
}


def _sql_type(dtype: pl.DataType) -> str:
    return _SQL_TYPES[dtype]



def _read(root: Path, kind: str, empty: pl.DataFrame, since: date | None) -> pl.DataFrame:
    files = sorted((root / kind).glob("*/part-*.parquet"))
    if not files:
        return empty
    pattern = (root / kind / "*" / "*.parquet").as_posix().replace("'", "''")
    where = ""
    params: list[object] = []
    if since is not None:
        where = " WHERE hour_utc >= ?"
        params.append(datetime.combine(since, datetime.min.time(), tzinfo=UTC))
    # hive_partitioning off keeps the stored `city` column and its plain dtype.
    sql = (
        f"SELECT * FROM read_parquet('{pattern}', hive_partitioning=false){where}"
        " ORDER BY city, hour_utc"
    )
    with duckdb.connect() as con:
        con.execute("SET TimeZone='UTC'")
        return fetch(con, sql, params)


def read_curated(root: Path = DEFAULT_ROOT, since: date | None = None) -> pl.DataFrame:
    frame = _read(root, "curated", schema.empty_curated(), since)
    return schema.conform(frame) if not frame.is_empty() else frame


def read_weather(root: Path = DEFAULT_ROOT, since: date | None = None) -> pl.DataFrame:
    return _read(root, "weather", schema.empty_weather(), since)


def prune(
    root: Path = DEFAULT_ROOT,
    keep_days: int = RETENTION_DAYS,
    today: date | None = None,
) -> list[Path]:
    today = today or date.today()
    cutoff = today - timedelta(days=keep_days)
    removed: list[Path] = []
    for kind in ("curated", "weather"):
        for path in sorted((root / kind).glob("*/part-*.parquet")):
            monthend = _month_end(path.stem.removeprefix("part-"))
            # The month holding today always survives: its end is never < cutoff.
            if monthend < cutoff:
                path.unlink()
                removed.append(path)
    return removed


def duck_conn(root: Path = DEFAULT_ROOT) -> duckdb.DuckDBPyConnection:
    """Connection with `curated` and `weather` views over the parquet store."""
    con = duckdb.connect()
    con.execute("SET TimeZone='UTC'")
    for kind, contract in (
        ("curated", schema.CURATED_SCHEMA),
        ("weather", schema.WEATHER_SCHEMA),
    ):
        if sorted((root / kind).glob("*/part-*.parquet")):
            pattern = (root / kind / "*" / "*.parquet").as_posix().replace("'", "''")
            con.execute(
                f"CREATE VIEW {kind} AS SELECT * FROM "
                f"read_parquet('{pattern}', hive_partitioning=false)"
            )
        else:
            # Typed empty view so the report SQL runs on a fresh checkout.
            cols = ", ".join(
                f"CAST(NULL AS {_sql_type(dtype)}) AS {name}"
                for name, dtype in contract.items()
            )
            con.execute(f"CREATE VIEW {kind} AS SELECT {cols} WHERE false")
    return con
