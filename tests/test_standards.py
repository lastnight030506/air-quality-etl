"""Benchmarks. The date-conditional QCVN PM2.5 value is the assertion that matters."""

from __future__ import annotations

from datetime import date

import pytest

from aqetl import standards


def test_pm25_24h_limit_before_2026_is_50():
    assert standards.qcvn_24h("pm25", date(2025, 12, 31)) == 50.0


def test_pm25_24h_limit_from_2026_is_45():
    """QCVN 05:2023 lists 50 with an asterisked 45 applying from 01/01/2026."""
    assert standards.qcvn_24h("pm25", date(2026, 1, 1)) == 45.0
    assert standards.qcvn_24h("pm25", date(2026, 11, 16)) == 45.0


@pytest.mark.parametrize(
    ("parameter", "expected"),
    [("pm10", 100.0), ("no2", 100.0), ("so2", 125.0)],
)
def test_qcvn_24h_table(parameter, expected):
    assert standards.qcvn_24h(parameter, date(2026, 11, 16)) == expected


def test_qcvn_does_not_regulate_24h_ozone():
    assert standards.qcvn_24h("o3", date(2026, 11, 16)) is None


@pytest.mark.parametrize(
    ("parameter", "expected"),
    [("pm25", 15.0), ("pm10", 45.0), ("no2", 25.0)],
)
def test_who_24h(parameter, expected):
    assert standards.who_24h(parameter) == expected


def test_who_annual_pm25_is_five():
    assert standards.limit_for("who", "pm25", "annual", date(2026, 11, 16)) == 5.0


def test_limit_for_unknown_standard_is_none():
    assert standards.limit_for("epa", "pm25", "24h", date(2026, 11, 16)) is None


def test_exceedance_ratio():
    assert standards.exceedance_ratio(90.0, 45.0) == 2.0
    assert standards.exceedance_ratio(None, 45.0) is None
    assert standards.exceedance_ratio(90.0, None) is None
    assert standards.exceedance_ratio(90.0, 0.0) is None


def test_vietnam_is_stricter_than_who_nowhere():
    """Sanity check on the reading of both documents: WHO is the tighter one."""
    day = date(2026, 11, 16)
    for parameter in ("pm25", "pm10", "no2"):
        assert standards.who_24h(parameter) < standards.qcvn_24h(parameter, day)


def test_benchmarks_for_includes_both_standards():
    names = {b.standard for b in standards.benchmarks_for("pm25", date(2026, 11, 16))}
    assert names == {"QCVN 05:2023", "WHO 2021"}
