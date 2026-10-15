"""RUL targets for documented run-to-failure units.

RUL(t) = documented failure time - current observation time.

The failure time is used only to build retrospective training/evaluation labels.
It is never an inference input: at prediction time the future failure time is
unknown, so any feature derived from it would be leakage.
Censored units (event not documented) receive no labels.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import ID, TVAL


def add_rul(obs: pd.DataFrame, units: pd.DataFrame, cap: float | None = None) -> pd.DataFrame:
    """Return obs with `rul` (physical, uncapped) and `rul_target` (possibly capped).

    Rows of censored or excluded units are dropped: their remaining life is unknown.
    """
    keep = units[units["include"].astype(bool) & units["event_observed"].astype(bool)]
    ft = keep.set_index(ID)["failure_time"].astype(float)
    out = obs[obs[ID].isin(ft.index)].copy()
    out["rul"] = out[ID].map(ft) - out[TVAL].astype(float)
    if (out["rul"] < 0).any():
        raise ValueError("negative RUL: an observation lies after the documented failure time")
    out["rul_target"] = np.minimum(out["rul"], cap) if cap is not None else out["rul"]
    return out


def censored_units(units: pd.DataFrame) -> list[str]:
    return units.loc[~units["event_observed"].astype(bool), ID].astype(str).tolist()
