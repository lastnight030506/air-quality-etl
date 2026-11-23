"""Storage, reporting and charting: DuckDB over Parquet."""

from __future__ import annotations

import polars as pl

from aqetl import chart, storage


def test_report_and_chart_end_to_end(tmp_path):
    """The README deliverable must exist even with no station data at all."""
    out = chart.render(root=tmp_path, out=tmp_path / "chart.png")
    assert out.exists()
    assert out.stat().st_size > 1000


def test_read_curated_on_missing_root_returns_empty(tmp_path):
    df = storage.read_curated(root=tmp_path)
    assert df.is_empty()


def test_write_then_read_roundtrip(tmp_path):
    from datetime import datetime

    from aqetl import schema

    row = {
        "city": "hanoi",
        "station_id": 1,
        "station_name": "Test",
        "sensor_id": 9,
        "parameter": "pm25",
        "hour_utc": datetime(2026, 11, 1, 0, 0),
        "value_raw": 12.0,
        "unit": "µg/m³",
        "unit_factor": 1.0,
        "value_ugm3": 12.0,
        "coverage_pct": 100.0,
        "qc_negative": False,
        "qc_impossible": False,
        "qc_stuck": False,
        "qc_spike": False,
        "qc_missing": False,
        "qc_flags": 0,
        "value_clean": 12.0,
    }
    df = pl.DataFrame([row], schema=schema.CURATED_SCHEMA)
    storage.write_curated(df, root=tmp_path)
    back = storage.read_curated(root=tmp_path)
    assert len(back) == 1
    assert back["city"].to_list() == ["hanoi"]


def test_write_is_idempotent(tmp_path):
    """Same frame twice must not double the row count.

    This is the property that keeps a twice-daily cron from accumulating
    duplicates of every hour it re-fetches.
    """
    from datetime import datetime, timedelta

    from aqetl import schema

    rows = []
    for i in range(3):
        rows.append(
            {
                "city": "hanoi",
                "station_id": 1,
                "station_name": "Test",
                "sensor_id": 9,
                "parameter": "pm25",
                "hour_utc": datetime(2026, 11, 1) + timedelta(hours=i),
                "value_raw": 10.0 + i,
                "unit": "µg/m³",
                "unit_factor": 1.0,
                "value_ugm3": 10.0 + i,
                "coverage_pct": 100.0,
                "qc_negative": False,
                "qc_impossible": False,
                "qc_stuck": False,
                "qc_spike": False,
                "qc_missing": False,
                "qc_flags": 0,
                "value_clean": 10.0 + i,
            }
        )
    df = pl.DataFrame(rows, schema=schema.CURATED_SCHEMA)
    storage.write_curated(df, root=tmp_path)
    first = len(storage.read_curated(root=tmp_path))
    storage.write_curated(df, root=tmp_path)
    second = len(storage.read_curated(root=tmp_path))
    assert first == second == 3


def test_duckdb_view_reads_partitions(tmp_path):
    """duck_conn must expose a queryable view over the parquet files."""
    from datetime import datetime

    from aqetl import schema, storage

    row = {
        "city": "hanoi", "station_id": 1, "station_name": "Test", "sensor_id": 9,
        "parameter": "pm25", "hour_utc": datetime(2026, 11, 1, 0, 0),
        "value_raw": 12.0, "unit": "µg/m³", "unit_factor": 1.0, "value_ugm3": 12.0,
        "coverage_pct": 100.0, "qc_negative": False, "qc_impossible": False,
        "qc_stuck": False, "qc_spike": False, "qc_missing": False, "qc_flags": 0,
        "value_clean": 12.0,
    }
    storage.write_curated(pl.DataFrame([row], schema=schema.CURATED_SCHEMA), root=tmp_path)

    con = storage.duck_conn(tmp_path)
    try:
        total = con.execute("SELECT count(*) FROM curated").fetchone()[0]
        city = con.execute("SELECT city FROM curated").fetchone()[0]
    finally:
        con.close()
    assert total == 1
    assert city == "hanoi"


def test_empty_store_view_still_answers(tmp_path):
    """A fresh checkout has no parquet files; report SQL must still run."""
    from aqetl import storage

    con = storage.duck_conn(tmp_path)
    try:
        assert con.execute("SELECT count(*) FROM curated").fetchone()[0] == 0
    finally:
        con.close()
