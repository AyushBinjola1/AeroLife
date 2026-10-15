"""Causal preprocessing fitted on training units only.

Every transform uses only (a) statistics fitted on training units and
(b) the same unit's present and past observations. Nothing looks ahead:
forward-fill only, trailing rolling means only, fixed scaling statistics.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .schema import ID, TIDX, TVAL

ELAPSED = "elapsed_time"


@dataclass
class CausalPreprocessor:
    channels: list[str]
    rolling_window: int = 0
    constant_std_threshold: float = 1e-8
    include_elapsed_time: bool = True
    fill_values: dict[str, float] = field(default_factory=dict)
    kept: list[str] = field(default_factory=list)
    dropped_constant: list[str] = field(default_factory=list)
    mean: dict[str, float] = field(default_factory=dict)
    std: dict[str, float] = field(default_factory=dict)
    fitted_on_units: list[str] = field(default_factory=list)

    # ---- core causal steps -------------------------------------------------
    def _causal(self, obs: pd.DataFrame) -> pd.DataFrame:
        df = obs.sort_values([ID, TIDX]).reset_index(drop=True)
        out = df[[ID, TIDX, TVAL]].copy()
        g = df.groupby(ID, sort=False)
        for c in self.channels:
            s = g[c].ffill()  # past-only imputation
            s = s.fillna(self.fill_values.get(c, 0.0))  # leading gaps: training median
            if self.rolling_window and self.rolling_window > 1:
                s = s.groupby(df[ID], sort=False).transform(
                    lambda x: x.rolling(self.rolling_window, min_periods=1).mean()
                )
            out[c] = s.astype(float)
        if self.include_elapsed_time:
            out[ELAPSED] = df[TVAL].astype(float)
        return out

    @property
    def feature_names(self) -> list[str]:
        return self.kept + ([ELAPSED] if self.include_elapsed_time else [])

    def fit(self, train_obs: pd.DataFrame) -> "CausalPreprocessor":
        self.fitted_on_units = sorted(train_obs[ID].astype(str).unique())
        self.fill_values = {c: float(np.nanmedian(train_obs[c])) if train_obs[c].notna().any() else 0.0
                            for c in self.channels}
        z = self._causal(train_obs)
        stds = z[self.channels].std(ddof=0)
        self.dropped_constant = [c for c in self.channels if not stds[c] > self.constant_std_threshold]
        self.kept = [c for c in self.channels if c not in self.dropped_constant]
        feats = self.feature_names
        self.mean = {c: float(z[c].mean()) for c in feats}
        self.std = {c: float(z[c].std(ddof=0)) or 1.0 for c in feats}
        return self

    def transform(self, obs: pd.DataFrame) -> pd.DataFrame:
        if not self.mean:
            raise RuntimeError("preprocessor is not fitted")
        z = self._causal(obs)
        for c in self.feature_names:
            z[c] = (z[c] - self.mean[c]) / self.std[c]
        return z[[ID, TIDX, TVAL, *self.feature_names]]

    # ---- serialisation (JSON, no pickle) ---------------------------------
    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, d: dict) -> "CausalPreprocessor":
        return cls(**d)


def audit(obs: pd.DataFrame, channels: list[str]) -> pd.DataFrame:
    """Per-channel audit: missing fraction, constancy, range."""
    rows = []
    for c in channels:
        s = obs[c]
        rows.append({
            "channel": c,
            "missing_fraction": float(s.isna().mean()),
            "std": float(s.std(ddof=0)),
            "min": float(s.min()),
            "max": float(s.max()),
            "n_unique": int(s.nunique()),
        })
    return pd.DataFrame(rows)
