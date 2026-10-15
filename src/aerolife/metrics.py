"""Evaluation metrics. All errors are in the dataset's own time unit.

error e = predicted RUL - true RUL  (e > 0: overestimation, i.e. predicting more
life than remains, the unsafe direction for maintenance).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import ID


def mae(e: np.ndarray) -> float:
    return float(np.mean(np.abs(e))) if len(e) else float("nan")


def rmse(e: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(e)))) if len(e) else float("nan")


def phm08_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Asymmetric scoring function of Saxena et al., PHM08 (2008):
    d = RUL_pred - RUL_true;  s = sum(exp(-d/13) - 1) for d < 0, sum(exp(d/10) - 1) for d >= 0.
    Defined for C-MAPSS cycles; its constants have no meaning in other time units,
    so it is only reported when the dataset is measured in cycles."""
    d = np.asarray(y_pred, float) - np.asarray(y_true, float)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13.0) - 1, np.exp(d / 10.0) - 1)))


def summarize(pred: pd.DataFrame, true_col: str = "rul", pred_col: str = "pred",
              near_failure: float | None = None) -> dict:
    """Pooled (window-weighted) and unit-averaged metrics.

    Pooled metrics weight each window equally, so long-lived units dominate.
    Unit-averaged metrics compute MAE/RMSE per unit first, then average units
    equally; they are the primary figure because units are the independent samples.
    """
    e = (pred[pred_col] - pred[true_col]).to_numpy()
    per_unit = per_unit_table(pred, true_col, pred_col)
    out = {
        "n_units": int(pred[ID].nunique()),
        "n_windows": int(len(pred)),
        "unit_avg_mae": float(per_unit["mae"].mean()),
        "unit_avg_rmse": float(per_unit["rmse"].mean()),
        "pooled_mae": mae(e),
        "pooled_rmse": rmse(e),
        "overestimate_fraction": float(np.mean(e > 0)) if len(e) else float("nan"),
        "mean_overestimate": float(e[e > 0].mean()) if (e > 0).any() else 0.0,
        "mean_underestimate": float(-e[e < 0].mean()) if (e < 0).any() else 0.0,
    }
    if near_failure is not None:
        near = pred[true_col].to_numpy() <= near_failure
        out["near_failure_mae"] = mae(e[near])
        out["early_life_mae"] = mae(e[~near])
        out["near_failure_windows"] = int(near.sum())
    return out


def per_unit_table(pred: pd.DataFrame, true_col: str = "rul", pred_col: str = "pred") -> pd.DataFrame:
    rows = []
    for uid, g in pred.groupby(ID):
        e = (g[pred_col] - g[true_col]).to_numpy()
        rows.append({ID: uid, "n_windows": len(g), "mae": mae(e), "rmse": rmse(e),
                     "bias": float(e.mean()), "overestimate_fraction": float(np.mean(e > 0))})
    return pd.DataFrame(rows).sort_values("mae", ascending=False).reset_index(drop=True)


def unit_bootstrap_ci(pred: pd.DataFrame, n_boot: int, seed: int, true_col: str = "rul",
                      pred_col: str = "pred", level: float = 0.95) -> dict:
    """Percentile CI of the unit-averaged MAE by resampling whole units.
    With few units the interval is wide and itself unreliable; report it as such."""
    per_unit = per_unit_table(pred, true_col, pred_col)["mae"].to_numpy()
    rng = np.random.default_rng(seed)
    if len(per_unit) < 2:
        return {"low": float("nan"), "high": float("nan"), "n_units": len(per_unit)}
    boots = rng.choice(per_unit, size=(n_boot, len(per_unit)), replace=True).mean(1)
    a = (1 - level) / 2
    return {"low": float(np.quantile(boots, a)), "high": float(np.quantile(boots, 1 - a)),
            "n_units": len(per_unit)}
