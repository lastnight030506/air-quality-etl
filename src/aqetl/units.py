"""Unit normalisation.

OpenAQ reports in whatever unit the upstream provider uses. PM is normally
already ug/m3; the gases are often ppm or ppb. Everything downstream assumes
ug/m3 at 25 C and 1 atm, which is the reference state QCVN 05:2023 uses (Nm3).
"""

from __future__ import annotations

import polars as pl

# Molar mass (g/mol) -> ug/m3 per ppm at 25 C, 1 atm (molar volume 24.45 L/mol).
_MOLAR_MASS = {
    "no2": 46.0055,
    "o3": 48.00,
    "so2": 64.066,
    "co": 28.010,
}

_MOLAR_VOLUME_L = 24.45


def ppm_to_ugm3(parameter: str, ppm: float) -> float:
    return ppm * _MOLAR_MASS[parameter] * 1000.0 / _MOLAR_VOLUME_L


def ugm3_per_ppm(parameter: str) -> float:
    """Conversion factor, exposed for tests and the README table."""
    return _MOLAR_MASS[parameter] * 1000.0 / _MOLAR_VOLUME_L


_UNIT_FACTORS = {
    # (parameter, unit) -> multiply value by this to reach ug/m3
    ("pm10", "µg/m³"): 1.0,
    ("pm25", "µg/m³"): 1.0,
    ("no2", "µg/m³"): 1.0,
    ("o3", "µg/m³"): 1.0,
    ("pm10", "ug/m3"): 1.0,
    ("pm25", "ug/m3"): 1.0,
    ("no2", "ug/m3"): 1.0,
    ("o3", "ug/m3"): 1.0,
    # ppm group filled in below so the factors stay in one place
}


def factor(parameter: str, unit: str) -> float | None:
    """Multiplier from `unit` to ug/m3, or None if the unit is unknown."""
    if unit is None:
        return None
    unit = unit.strip()
    if (parameter, unit) in _UNIT_FACTORS:
        return _UNIT_FACTORS[(parameter, unit)]
    lowered = unit.lower().replace("μ", "µ")
    if lowered in {"µg/m³", "ug/m3", "µg/m3", "ug/m³"}:
        return 1.0
    # A gas unit on a particulate parameter is not convertible by molar mass.
    # Return None (a recoverable null) rather than letting _MOLAR_MASS raise and
    # take the whole run down with it.
    if parameter not in _MOLAR_MASS:
        return None
    if lowered in {"ppb", "parts per billion"}:
        return ugm3_per_ppm(parameter) / 1000.0
    if lowered in {"ppm", "parts per million"}:
        return ugm3_per_ppm(parameter)
    return None


def to_ugm3(df: pl.DataFrame, value_col: str = "value_raw", unit_col: str = "unit") -> pl.DataFrame:
    """Adds `value_ugm3` and `unit_factor`; unknown units become null.

    Null rather than a guess: a wrong conversion silently corrupts every
    exceedance statistic that follows.
    """
    if df.is_empty():
        return df.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("unit_factor"),
            pl.lit(None, dtype=pl.Float64).alias("value_ugm3"),
        )
    pairs = df.select(pl.col("parameter"), pl.col(unit_col)).unique()
    factors = pl.DataFrame(
        {
            "parameter": pairs["parameter"],
            unit_col: pairs[unit_col],
            "unit_factor": [
                factor(p, u) for p, u in zip(pairs["parameter"], pairs[unit_col], strict=True)
            ],
        },
        schema_overrides={"unit_factor": pl.Float64()},
    )
    # A join keeps this vectorised; a per-row Python callback both hides a
    # mis-keyed lookup and costs a row-wise pass over the whole frame.
    return (
        df.join(factors, on=["parameter", unit_col], how="left")
        .with_columns((pl.col(value_col) * pl.col("unit_factor")).alias("value_ugm3"))
    )
