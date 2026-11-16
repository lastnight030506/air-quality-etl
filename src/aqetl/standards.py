"""Regulatory and health benchmarks.

Two references are encoded:

* QCVN 05:2023/BTNMT (Vietnam national technical regulation on ambient air
  quality), Table 1, in ug/Nm3.
* WHO global air quality guidelines, 2021 update, in ug/m3.

Both are monthly/24h oriented. The pipeline compares 24h means, because that is
the shortest averaging period both documents regulate for PM.

Note on QCVN PM2.5: the 24h cell carries two numbers. The regulation lists 50
ug/Nm3 with an asterisked 45 that applies from 01/01/2026. Since we are past
that date, 45 is the binding value; the earlier value is kept for dates before
it so historical runs stay faithful.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

# QCVN 05:2023/BTNMT, Table 1. Averaging period -> limit in ug/Nm3.
QCVN_1H: dict[str, float] = {"so2": 350, "no2": 200, "o3": 200, "co": 30_000}
QCVN_8H: dict[str, float] = {"co": 10_000, "o3": 120}
QCVN_24H: dict[str, float] = {
    "so2": 125,
    "no2": 100,
    "pm10": 100,
    "tsp": 200,
}
QCVN_ANNUAL: dict[str, float] = {
    "so2": 50,
    "no2": 40,
    "pm10": 50,
    "tsp": 100,
}

# PM2.5 is date-conditional, so it is not a plain dict entry.
QCVN_PM25_24H_BEFORE_2026 = 50.0
QCVN_PM25_24H_FROM_2026 = 45.0
QCVN_PM25_ANNUAL_BEFORE_2026 = 25.0
QCVN_PM25_ANNUAL_FROM_2026 = 25.0
QCVN_PM25_CHANGE_DATE = date(2026, 1, 1)

# WHO 2021 AQG levels.
WHO_24H: dict[str, float] = {"pm25": 15.0, "pm10": 45.0, "no2": 25.0, "so2": 40.0}
WHO_ANNUAL: dict[str, float] = {"pm25": 5.0, "pm10": 15.0, "no2": 10.0}
WHO_8H: dict[str, float] = {"o3": 100.0}
WHO_1H: dict[str, float] = {"no2": 200.0}


@dataclass(frozen=True)
class Benchmark:
    standard: str
    parameter: str
    averaging: str
    limit_ugm3: float
    source: str


def qcvn_24h(parameter: str, on: date) -> float | None:
    """QCVN 24h limit for a parameter on a given date, or None if unregulated."""
    if parameter == "pm25":
        if on < QCVN_PM25_CHANGE_DATE:
            return QCVN_PM25_24H_BEFORE_2026
        return QCVN_PM25_24H_FROM_2026
    return QCVN_24H.get(parameter)


def qcvn_annual(parameter: str, on: date) -> float | None:
    if parameter == "pm25":
        if on < QCVN_PM25_CHANGE_DATE:
            return QCVN_PM25_ANNUAL_BEFORE_2026
        return QCVN_PM25_ANNUAL_FROM_2026
    return QCVN_ANNUAL.get(parameter)


def who_24h(parameter: str) -> float | None:
    return WHO_24H.get(parameter)


def qcvn_8h(parameter: str) -> float | None:
    return QCVN_8H.get(parameter)


def who_8h(parameter: str) -> float | None:
    return WHO_8H.get(parameter)


def limit_for(standard: str, parameter: str, averaging: str, on: date) -> float | None:
    """Single dispatch point so callers never hardcode a benchmark."""
    standard = standard.lower()
    if standard == "qcvn":
        if averaging == "24h":
            return qcvn_24h(parameter, on)
        if averaging == "annual":
            return qcvn_annual(parameter, on)
        if averaging == "8h":
            return qcvn_8h(parameter)
        if averaging == "1h":
            return QCVN_1H.get(parameter)
    elif standard == "who":
        if averaging == "24h":
            return who_24h(parameter)
        if averaging == "annual":
            return WHO_ANNUAL.get(parameter)
        if averaging == "8h":
            return who_8h(parameter)
        if averaging == "1h":
            return WHO_1H.get(parameter)
    return None


def exceedance_ratio(value: float | None, limit: float | None) -> float | None:
    if value is None or limit is None or limit <= 0:
        return None
    return value / limit


def benchmarks_for(parameter: str, on: date) -> list[Benchmark]:
    """Every benchmark that applies to a parameter on a date, for reporting."""
    out: list[Benchmark] = []
    q = qcvn_24h(parameter, on)
    if q is not None:
        out.append(Benchmark("QCVN 05:2023", parameter, "24h", q, "QCVN 05:2023/BTNMT Table 1"))
    w = who_24h(parameter)
    if w is not None:
        out.append(
            Benchmark("WHO 2021", parameter, "24h", w, "WHO global air quality guidelines 2021")
        )
    return out
