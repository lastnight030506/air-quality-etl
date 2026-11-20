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
from .config import CITIES, OPENAQ_BASE, PARAMETERS, City
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
    if raw.is_empty():
        # Committing empty reports would look like a healthy run and erase the
        # evidence that the extract failed. Fail loudly instead.
        print(
            "aqetl run: extract returned no rows. Either the API key lacks access, "
            "no station in range reports the configured parameters, or the query "
            "shape is wrong. Nothing was written.",
            file=sys.stderr,
        )
        return 1

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


def _discover(args: argparse.Namespace) -> int:
    """Rank Vietnamese localities by how recently a station there reported.

    Written because the brief named three cities and only two turned out to have
    live OpenAQ stations; the coverage has to be measured, not assumed.
    """
    from collections import Counter, defaultdict

    import httpx

    from .openaq import _request

    cutoff = (datetime.now(tz=UTC) - timedelta(days=args.days)).strftime("%Y-%m-%dT%H:%M:%SZ")
    localities: Counter[str] = Counter()
    newest: dict[str, str] = {}
    params: dict[str, set[str]] = defaultdict(set)
    all_last: list[str] = []
    fetched = 0
    page = 1
    with httpx.Client(timeout=30.0) as client:
        while page <= 20:
            payload = _request(
                client,
                f"{OPENAQ_BASE}/locations",
                {"iso": "VN", "limit": 1000, "page": page},
            )
            batch = payload.get("results") or []
            fetched += len(batch)
            for loc in batch:
                last = (loc.get("datetimeLast") or {}).get("utc") or ""
                all_last.append(last)
                if last < cutoff:
                    continue
                name = (loc.get("locality") or "unknown").strip()
                localities[name] += 1
                if last > newest.get(name, ""):
                    newest[name] = last
                for sensor in loc.get("sensors") or []:
                    pid = (sensor.get("parameter") or {}).get("id")
                    if pid in PARAMETERS:
                        params[name].add(PARAMETERS[pid])
            if len(batch) < 1000:
                break
            page += 1

    print(f"localities with a station reporting in the last {args.days} days:")
    for name, count in localities.most_common():
        print(
            "LOC|{}|{}|{}|{}".format(
                name.replace("|", "/") or "unknown",
                count,
                newest[name],
                ",".join(sorted(params[name])) or "none",
            )
        )
    print(f"  [diag] pages={page} fetched={fetched} recent={sum(localities.values())} "
          f"max_datetimeLast={max((x for x in all_last if x), default='none')}")
    return 0


def _report() -> int:
    paths = report.write_report(ROOT)
    for name, path in paths.items():
        _log(name, str(path))
    _log("chart", str(chart.render(ROOT, CHART)))
    return 0


def _probe(args: argparse.Namespace) -> int:
    """Print what the APIs actually return.

    Written because a zero-row run is otherwise indistinguishable from a healthy
    one, and the response shape cannot be observed without the key.
    """
    import json
    from datetime import timedelta
    from time import perf_counter

    import httpx

    from .openaq import _request, fetch_locations
    from .openmeteo import ARCHIVE_URL, FORECAST_URL

    print("== network ==")
    for label, url in (("archive", ARCHIVE_URL), ("forecast", FORECAST_URL)):
        started = perf_counter()
        try:
            response = httpx.get(
                url,
                params={"latitude": 21.0285, "longitude": 105.8542,
                        "hourly": "temperature_2m", "past_days": 1},
                timeout=20.0,
            )
            print(f"  {label:9} {response.status_code} in {perf_counter() - started:.2f}s")
        except Exception as exc:  # noqa: BLE001 - diagnostic
            print(f"  {label:9} FAILED in {perf_counter() - started:.2f}s "
                  f"{type(exc).__name__}: {exc}")

    # The bare-request test above passes in CI while the pipeline fails, so the
    # gap is somewhere in fetch_city. Exercise it exactly as the pipeline does.
    print("== openmeteo.fetch_all, as the pipeline calls it ==")
    from .openmeteo import fetch_all as fetch_meteo_all

    started = perf_counter()
    try:
        with httpx.Client(timeout=30.0) as client:
            frame = fetch_meteo_all(args.meteo_start, args.meteo_end)
        print(f"  ok rows={frame.height} in {perf_counter() - started:.2f}s "
              f"nulls={frame['temperature_2m'].null_count()} "
              f"cities={sorted(frame['city'].unique().to_list())}")
    except Exception as exc:  # noqa: BLE001 - diagnostic
        print(f"  FAILED in {perf_counter() - started:.2f}s "
              f"{type(exc).__name__}: {exc}")

    end = datetime.now(tz=UTC)
    with httpx.Client(timeout=30.0) as client:
        for city in args.cities:
            try:
                locations = fetch_locations(client, city)
            except Exception as exc:  # noqa: BLE001 - this command is diagnostic
                print(f"{city.key}: locations call failed: {exc}")
                continue
            with_target = []
            print(f"\n== {city.key} == locations in range: {len(locations)}")
            recency: list[tuple[str, str]] = []
            for loc in locations:
                sensors = [
                    (s.get("id"), (s.get("parameter") or {}).get("id"),
                     (s.get("parameter") or {}).get("units"))
                    for s in (loc.get("sensors") or [])
                ]
                hits = [s for s in sensors if s[1] in PARAMETERS]
                if hits:
                    with_target.append((loc, hits))
                    last = (loc.get("datetimeLast") or {}).get("utc") or "never"
                    recency.append((last, str(loc.get("name"))))
            print(f"  with a target parameter: {len(with_target)}")
            # The decisive question: does ANY station in this city still report?
            for last, name in sorted(recency, reverse=True)[:5]:
                print(f"  newest: {last}  {name[:48]}")
            if not with_target:
                continue

            # Newest station, not just the first: an old station would make the
            # window test meaningless.
            with_target.sort(
                key=lambda pair: (pair[0].get("datetimeLast") or {}).get("utc") or "", reverse=True
            )
            loc, hits = with_target[0]
            sensor_id, param_id, units = hits[0]
            print(
                f"  sample station id={loc.get('id')} {loc.get('name')!r} "
                f"sensor={sensor_id} units={units}"
            )
            # The station's own reported coverage tells us whether a zero result
            # means "no data in this window" or "this station stopped years ago".
            meta = _request(client, f"{OPENAQ_BASE}/sensors/{sensor_id}", {})
            result = (meta.get("results") or [{}])[0]
            print(f"  datetimeFirst={result.get('datetimeFirst')}")
            print(f"  datetimeLast={result.get('datetimeLast')}")
            print(f"  latest={json.dumps(result.get('latest'))[:200]}")
            for label, days in (("2d", 2), ("90d", 90), ("365d", 365)):
                window_start = datetime.now(tz=UTC) - timedelta(days=days)
                payload = _request(
                    client,
                    f"{OPENAQ_BASE}/sensors/{sensor_id}/hours",
                    {
                        "datetime_from": window_start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "datetime_to": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "limit": 100,
                    },
                )
                results = payload.get("results") or []
                print(
                    f"  hours[{label}] found={(payload.get('meta') or {}).get('found')}"
                    f" returned={len(results)}"
                )
                if results:
                    print("  first record:", json.dumps(results[0])[:420])
                    break
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

    probe = sub.add_parser("probe", help="print the raw OpenAQ station coverage per city")
    probe.add_argument("--cities", type=_cities, default=CITIES)
    probe.add_argument("--meteo-start", type=_parse_date,
                       default=date.today() - timedelta(days=90))
    probe.add_argument("--meteo-end", type=_parse_date, default=date.today())

    discover = sub.add_parser("discover", help="rank Vietnamese localities by live station count")
    discover.add_argument("--days", type=int, default=7)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run" and args.since > args.until:
        print("--since is after --until", file=sys.stderr)
        return 1
    try:
        if args.command == "run":
            return _run(args)
        if args.command == "probe":
            return _probe(args)
        if args.command == "discover":
            return _discover(args)
        return _report()
    except Exception as exc:  # the workflow wants a legible one-line failure
        print(f"aqetl {args.command}: {exc}", file=sys.stderr)
        return 1


__all__ = ["build_parser", "main"]
