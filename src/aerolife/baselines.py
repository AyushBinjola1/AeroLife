"""Training-derived baselines, evaluated on exactly the same windows as the LSTM."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge


class ConstantBaseline:
    """Predicts the median training target, whatever the input."""

    name = "constant_median"

    def fit(self, y_train: np.ndarray) -> "ConstantBaseline":
        self.value = float(np.median(y_train))
        return self

    def predict(self, n: int) -> np.ndarray:
        return np.full(n, self.value, dtype=float)


class AgeBaseline:
    """Empirical mean residual life given current age, from training lifetimes:

        pred(t) = mean_i (L_i - t)  over training units with L_i > t,  else 0

    where L_i is the documented failure time of training unit i. Uses only the
    current elapsed time, which is known at prediction time.
    """

    name = "age_mean_residual_life"

    def fit(self, train_lifetimes: np.ndarray, cap: float | None = None) -> "AgeBaseline":
        self.lifetimes = np.sort(np.asarray(train_lifetimes, dtype=float))
        self.cap = cap
        return self

    def predict(self, ages: np.ndarray) -> np.ndarray:
        out = np.zeros(len(ages))
        for i, t in enumerate(np.asarray(ages, dtype=float)):
            alive = self.lifetimes[self.lifetimes > t]
            out[i] = (alive - t).mean() if alive.size else 0.0
        return np.minimum(out, self.cap) if self.cap is not None else out


def make_ridge(alpha: float) -> Ridge:
    return Ridge(alpha=alpha)


def make_rf(n_estimators: int, max_depth: int | None, seed: int) -> RandomForestRegressor:
    return RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                 min_samples_leaf=5, n_jobs=-1, random_state=seed)
