"""Unit conversion. A wrong factor corrupts every downstream exceedance."""

from __future__ import annotations

import pytest

from aqetl import units


@pytest.mark.parametrize(
    ("parameter", "expected"),
    [
        ("no2", 1881.6),
        ("o3", 1963.2),
        ("so2", 2620.3),
        ("co", 1145.6),
    ],
)
def test_ppm_factor_matches_molar_mass_at_25c(parameter, expected):
    assert units.ugm3_per_ppm(parameter) == pytest.approx(expected, abs=0.1)


def test_microgram_units_are_identity():
    assert units.factor("pm25", "µg/m³") == 1.0
    assert units.factor("pm25", "ug/m3") == 1.0


def test_ppb_is_a_thousandth_of_ppm():
    assert units.factor("no2", "ppb") == pytest.approx(units.ugm3_per_ppm("no2") / 1000)


def test_unknown_unit_returns_none_rather_than_guessing():
    """Null is recoverable; a wrong multiplier is silent corruption."""
    assert units.factor("pm25", "bananas") is None
    assert units.factor("pm25", None) is None


def test_to_ugm3_converts_mixed_units_per_parameter():
    import polars as pl

    df = pl.DataFrame(
        {
            "parameter": ["pm25", "no2"],
            "unit": ["µg/m³", "ppm"],
            "value_raw": [12.0, 0.1],
        }
    )
    out = units.to_ugm3(df)
    values = out["value_ugm3"].to_list()
    assert values[0] == 12.0
    assert values[1] == pytest.approx(188.16, abs=0.1)


def test_gas_unit_on_a_particulate_returns_none_not_a_crash():
    """REGRESSION: factor('pm25','ppm') raised KeyError and killed the run."""
    assert units.factor("pm25", "ppm") is None
    assert units.factor("pm10", "ppb") is None
