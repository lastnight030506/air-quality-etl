"""Command line entry point.

`run` spends API calls, `report` does not; the split exists so the README can be
refreshed from the committed store without touching a rate-limited free tier.
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path

from . import align, chart, qaqc, report, storage, units
from .config import CITIES, City
from .schema import QC_FLAG_COLUMNS, empty_weather

ROOT = Path("data")
CHART = Path("charts/aq-30d.png")


def _parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"not a YYYY-MM-DD date: {value}") from exc


def _cities(value: str) -> tuple[City, ...]:
    keys = [part.strip() for part in value.split(",") if part.strip()]
    known = {city.key: city for city in CITIES}
    unknown = [key for key in keys if key not in known]
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown city: {', '.join(unknown)} (choose from {', '.join(known)})"
        )
    return tuple(known[key] for key in keys)


def _log(stage: str, detail: str) -> None:
    print(f"{stage:<12} {detail}")


def _run(args: argparse.Namespace) -> int:
    # Imported here so `report` and the tests never need the extract layer.
    from .gaps import fill_gaps
    from .openaq import fetch_all
    from .openmeteo import fetch_all as fetch_weather

    start = datetime.combine(args.since, time.min, tzinfo=UTC)
    end = datetime.combine(args.until + timedelta(days=1), time.min, tzinfo=UTC)

    raw = fetch_all(start, end, cities=args.cities)
    _log("extract", f"{len(raw)} rows")

    # Weather is enrichment, not the deliverable: a met-service outage must not
    # discard observations we already paid API calls for.
    try:
        weather = fetch_weather(args.since, args.until, cities=args.cities)
        _log("weather", f"{len(weather)} rows")
    except Exception as exc:
        _log("weather", f"unavailable ({exc.__class__.__name__}: {exc}) — continuing")
        weather = empty_weather()

    observed = units.to_ugm3(raw)
    observed = fill_gaps(observed, start, end)
    _log("gaps", f"{len(observed)} rows")

    curated = qaqc.apply_all(observed)
    counts = ", ".join(
        f"{name.removeprefix('qc_')}={curated[name].sum()}" for name in QC_FLAG_COLUMNS
    )
    _log("qc", f"{len(curated)} rows ({counts})")

    joined = align.join_weather(curated, weather)
    _log("align", f"{len(joined)} rows")

    if args.dry_run:
        # Report is an aggregation step, so dry-run can still show its counts
        # without writing the frames it would otherwise persist.
        for name, frame in (
            ("daily", report.build_daily(ROOT, 30)),
            ("station_qc", report.build_station_qc(ROOT, 30)),
            ("exceedance", report.build_exceedance(ROOT, 90)),
        ):
            _log(name, f"{frame.height} rows")
        _log("dry-run", "nothing written")
        return 0

    paths = storage.write_curated(joined, ROOT)
    _log("load", f"{len(joined)} rows to {len(paths)} files")
    storage.write_weather(weather, ROOT)
    _log("prune", f"{len(storage.prune(ROOT))} files removed")
    _report()
    return 0


def _report() -> int:
    paths = report.write_report(ROOT)
    for name, path in paths.items():
        _log(name, str(path))
    _log("chart", str(chart.render(ROOT, CHART)))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aqetl", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="fetch, transform, load, report")
    run.add_argument("--since", type=_parse_date, default=date.today() - timedelta(days=30))
    run.add_argument("--until", type=_parse_date, default=date.today())
    run.add_argument("--cities", type=_cities, default=CITIES)
    run.add_argument("--dry-run", action="store_true")

    sub.add_parser("report", help="rebuild reports and chart from stored parquet")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run" and args.since > args.until:
        print("--since is after --until", file=sys.stderr)
        return 1
    try:
        return _run(args) if args.command == "run" else _report()
    except Exception as exc:  # the workflow wants a legible one-line failure
        print(f"aqetl {args.command}: {exc}", file=sys.stderr)
        return 1


__all__ = ["build_parser", "main"]
