"""Sliding windows built independently inside each unit of one partition.

A window of length L ending at step e uses observations e-L+1 .. e of a single
unit and is labelled with the RUL at step e (its last observed step). Ends are
anchored to the unit's final observation and step back by `stride`.
Units with fewer than L observations yield no windows ("insufficient history");
nothing is padded.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import ID, TIDX, TVAL


def window_ends(n: int, length: int, stride: int) -> np.ndarray:
    if length < 1 or stride < 1:
        raise ValueError("window length and stride must be positive")
    if n < length:
        return np.array([], dtype=int)
    return np.arange(n - 1, length - 2, -stride)[::-1]


def make_windows(
    df: pd.DataFrame,
    feature_cols: list[str],
    length: int,
    stride: int = 1,
    target_col: str | None = None,
    extra_meta: tuple[str, ...] = ("rul",),
):
    """Return X (N, L, F), y (N,) or None, and a meta DataFrame per window."""
    xs, ys, meta = [], [], []
    for uid, g in df.groupby(ID, sort=True):
        g = g.sort_values(TIDX)
        vals = g[feature_cols].to_numpy(dtype=np.float32)
        ends = window_ends(len(g), length, stride)
        indices = g[TIDX].to_numpy()
        for e in ends:
            if not np.all(np.diff(indices[e - length + 1 : e + 1]) == 1):
                continue  # missing recordings cannot be treated as consecutive time steps
            xs.append(vals[e - length + 1 : e + 1])
            row = {ID: uid, TIDX: int(g[TIDX].iloc[e]), TVAL: float(g[TVAL].iloc[e]),
                   "window_start_index": int(g[TIDX].iloc[e - length + 1])}
            for m in extra_meta:
                if m in g.columns:
                    row[m] = float(g[m].iloc[e])
            meta.append(row)
            if target_col:
                ys.append(float(g[target_col].iloc[e]))
    X = np.stack(xs) if xs else np.zeros((0, length, len(feature_cols)), dtype=np.float32)
    y = np.asarray(ys, dtype=np.float32) if target_col else None
    return X, y, pd.DataFrame(meta)


def summarize_windows(X: np.ndarray, feature_names: list[str]) -> tuple[np.ndarray, list[str]]:
    """Fixed-length summary of the same window history for tabular baselines:
    last value, mean, std, min, max and least-squares slope per feature."""
    n, L, F = X.shape
    t = np.arange(L, dtype=np.float32)
    tc = t - t.mean()
    denom = float((tc**2).sum()) or 1.0
    slope = np.einsum("l,nlf->nf", tc, X) / denom
    parts = [X[:, -1, :], X.mean(1), X.std(1), X.min(1), X.max(1), slope]
    names = [f"{s}_{f}" for s in ("last", "mean", "std", "min", "max", "slope") for f in feature_names]
    return np.concatenate(parts, axis=1), names
