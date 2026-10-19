"""Canonical data schema and validation.

Every dataset adapter must emit two tables:

observations  one row per observed time step of one unit
    unit_id     str    independent equipment unit (engine, bearing, ...)
    time_index  int    0, 1, 2, ... order of observation within the unit
    time_value  float  elapsed time since the start of the unit's record, in the
                       dataset's documented time unit (cycles, minutes, ...)
    <channels>  float  measured channels / documented settings listed in the spec

units         one row per unit
    unit_id         str
    event_observed  bool   True only if the source documents the failure/end-of-life event
    failure_time    float  documented failure time (same unit as time_value); NaN if censored
    condition_id    str    documented operating condition (optional)
    include         bool
    exclusion_reason str

Label-like columns (RUL, failure time, health parameters, ...) are never
allowed as model inputs; they are detected by name below.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

ID, TIDX, TVAL = "unit_id", "time_index", "time_value"
OBS_REQUIRED = (ID, TIDX, TVAL)
UNITS_REQUIRED = (ID, "event_observed", "failure_time", "include", "exclusion_reason")

# Columns that encode the answer or post-failure / hidden-state information.
FORBIDDEN_INPUT_PATTERNS = [
    r"^rul",
    r"remaining",
    r"failure",
    r"^eol",
    r"end_of_life",
    r"event_observed",
    r"^health",
    r"^theta",
    r"^hs$",
    r"degradation_state",
    r"^true_",
    r"^target",
    r"^label",
    r"run_length",
    r"total_life",
    r"lifetime",
]


def is_forbidden_input(name: str) -> bool:
    n = name.strip().lower()
    return any(re.search(p, n) for p in FORBIDDEN_INPUT_PATTERNS)


@dataclass
class ValidationReport:
    ok: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    dropped_columns: list[str] = field(default_factory=list)

    def error(self, msg: str) -> None:
        self.ok = False
        self.errors.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)


def validate_observations(
    df: pd.DataFrame,
    channels: list[str] | None = None,
    allow_extra: bool = True,
) -> ValidationReport:
    """Structural checks on an observations table (also used for uploads)."""
    rep = ValidationReport()
    missing = [c for c in OBS_REQUIRED if c not in df.columns]
    if missing:
        rep.error(f"missing required columns: {missing}")
        return rep
    if df.empty:
        rep.error("table has no rows")
        return rep
    if df.columns.duplicated().any():
        rep.error("duplicate column names")
        return rep
    if df[ID].isna().any() or df[ID].astype(str).str.strip().eq("").any():
        rep.error("unit_id must not be missing or blank")
    if df[ID].map(type).nunique() > 1:
        rep.error("unit_id must use a consistent type")

    leaky = [c for c in df.columns if c not in OBS_REQUIRED and is_forbidden_input(c)]
    if leaky:
        rep.dropped_columns.extend(leaky)
        rep.warn(f"label-like columns are ignored and never used as inputs: {leaky}")

    if channels is not None:
        absent = [c for c in channels if c not in df.columns]
        if absent:
            rep.error(f"missing expected channels: {absent}")
        if not allow_extra:
            extra = [c for c in df.columns if c not in (*OBS_REQUIRED, *channels, *leaky)]
            if extra:
                rep.error(f"unexpected columns: {extra}")
        check_cols = [c for c in channels if c in df.columns]
    else:
        check_cols = [c for c in df.columns if c not in (*OBS_REQUIRED, *leaky)]

    for c in check_cols:
        if not pd.api.types.is_numeric_dtype(df[c]):
            rep.error(f"channel '{c}' is not numeric")
        elif np.isinf(df[c].to_numpy(dtype=float)).any():
            rep.error(f"channel '{c}' contains infinite values")
        else:
            frac = float(df[c].isna().mean())
            if frac > 0:
                rep.warn(f"channel '{c}' has {frac:.1%} missing values (past-only fill applies)")

    if not pd.api.types.is_integer_dtype(df[TIDX]):
        rep.error("time_index must be integer")
    if not pd.api.types.is_numeric_dtype(df[TVAL]):
        rep.error("time_value must be numeric")
    else:
        if not np.isfinite(df[TVAL].to_numpy(dtype=float, na_value=np.nan)).all():
            rep.error("time_value must contain finite values")
        if (df[TVAL] < 0).any():
            rep.error("time_value must be non-negative elapsed time")
    if df[TIDX].isna().any():
        rep.error("time_index must not be missing")
    elif pd.api.types.is_integer_dtype(df[TIDX]) and (df[TIDX] < 0).any():
        rep.error("time_index must be non-negative")
    if not rep.ok:
        return rep

    if df[[ID, TIDX]].duplicated().any():
        rep.error("duplicate (unit_id, time_index) rows")
    for uid, g in df.groupby(ID, sort=False):
        g = g.sort_values(TIDX)
        if not g[TVAL].is_monotonic_increasing:
            rep.error(f"unit {uid}: time_value is not non-decreasing in time_index order")
        if g[TVAL].isna().any():
            rep.error(f"unit {uid}: time_value has missing values")
        gaps = np.diff(g[TIDX].to_numpy())
        if (gaps > 1).any():
            rep.warn(f"unit {uid}: {int((gaps > 1).sum())} gap(s) in time_index (recording gaps)")
    return rep


def validate_units(units: pd.DataFrame, obs: pd.DataFrame | None = None) -> ValidationReport:
    rep = ValidationReport()
    missing = [c for c in UNITS_REQUIRED if c not in units.columns]
    if missing:
        rep.error(f"units table missing columns: {missing}")
        return rep
    if units[ID].duplicated().any():
        rep.error("duplicate unit_id in units table")
    ev = units["event_observed"].astype(bool)
    bad = units.loc[ev & units["failure_time"].isna(), ID].tolist()
    if bad:
        rep.error(f"units marked event_observed without failure_time: {bad}")
    cens = units.loc[~ev & units["failure_time"].notna(), ID].tolist()
    if cens:
        rep.error(f"censored units must not carry a failure_time (would invent life): {cens}")
    excl = units.loc[~units["include"].astype(bool) & (units["exclusion_reason"].fillna("") == ""), ID]
    if len(excl):
        rep.error(f"excluded units need an exclusion_reason: {excl.tolist()}")
    if obs is not None and rep.ok:
        last = obs.groupby(ID)[TVAL].max()
        for _, r in units[ev].iterrows():
            uid = r[ID]
            if uid not in last.index:
                rep.error(f"unit {uid} has no observations")
            elif r["failure_time"] < last[uid]:
                rep.error(f"unit {uid}: failure_time precedes the last observation")
        orphan = sorted(set(obs[ID]) - set(units[ID]))
        if orphan:
            rep.error(f"observations for units not in units table: {orphan}")
    return rep


def input_columns(df: pd.DataFrame, spec_channels: list[str]) -> list[str]:
    """Measured channels permitted as model inputs (never label-like)."""
    cols = [c for c in spec_channels if c in df.columns and not is_forbidden_input(c)]
    leaked = [c for c in spec_channels if is_forbidden_input(c)]
    if leaked:
        raise ValueError(f"dataset spec lists label-like columns as inputs: {leaked}")
    return cols
