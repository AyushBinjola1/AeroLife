import numpy as np
import pandas as pd

from aerolife.features import burst_features
from aerolife.metrics import phm08_score, summarize, unit_bootstrap_ci
from aerolife.preprocessing import CausalPreprocessor


def _fit(obs, train=("u1", "u2", "u3"), rolling=3):
    return CausalPreprocessor(["ch_a", "ch_b", "ch_const"], rolling_window=rolling).fit(
        obs[obs.unit_id.isin(train)])


def test_constant_channel_dropped_and_fit_on_train_only(tables):
    obs, _ = tables
    p = _fit(obs)
    assert p.dropped_constant == ["ch_const"] and p.fitted_on_units == ["u1", "u2", "u3"]
    altered = obs.copy(); altered.loc[altered.unit_id == "u5", "ch_a"] += 1e6
    assert _fit(altered).mean == p.mean  # held-out values never touch fitted statistics


def test_future_values_do_not_change_past_outputs(tables):
    obs, _ = tables
    p = _fit(obs)
    u = obs[obs.unit_id == "u4"].copy()
    base = p.transform(u)
    fut = u.copy(); fut.loc[fut.time_index > 20, ["ch_a", "ch_b"]] = 12345.0
    fut.loc[fut.time_index == 30, "ch_b"] = np.nan
    after = p.transform(fut)
    pd.testing.assert_frame_equal(base[base.time_index <= 20], after[after.time_index <= 20])


def test_past_only_fill(tables):
    obs, _ = tables
    p = _fit(obs, rolling=0)
    u = obs[obs.unit_id == "u4"].copy()
    u.loc[u.time_index == 10, "ch_a"] = np.nan
    z = p.transform(u)
    assert z.loc[z.time_index == 10, "ch_a"].item() == z.loc[z.time_index == 9, "ch_a"].item()


def test_serialisation_roundtrip(tables):
    obs, _ = tables
    p = _fit(obs)
    q = CausalPreprocessor.from_dict(p.to_dict())
    pd.testing.assert_frame_equal(p.transform(obs), q.transform(obs))


def test_burst_features_formulas():
    fs, n = 1000.0, 1000
    t = np.arange(n) / fs
    x = 2.0 * np.sin(2 * np.pi * 50 * t)
    f = burst_features(x, fs, bands_hz=[(40, 60), (100, 200)])
    assert abs(f["rms"] - 2 / np.sqrt(2)) < 1e-6
    assert abs(f["crest_factor"] - np.sqrt(2)) < 1e-3
    assert abs(f["kurtosis"] - 1.5) < 1e-3          # Pearson kurtosis of a sinusoid
    assert abs(f["spectral_centroid"] - 50) < 1e-6
    assert f["band_40_60"] > 0.999
    flat = burst_features(np.zeros(100), fs)
    assert all(np.isfinite(v) for v in flat.values()) and flat["crest_factor"] == 0.0


def test_phm08_score_sign_convention():
    assert phm08_score([10], [10]) == 0
    late, early = phm08_score([10], [20]), phm08_score([20], [10])   # over- vs under-estimate by 10
    assert np.isclose(late, np.e - 1) and np.isclose(early, np.exp(10 / 13) - 1) and late > early


def test_unit_average_vs_pooled():
    df = pd.DataFrame({"unit_id": ["a"] * 9 + ["b"], "rul": [0.0] * 10, "pred": [1.0] * 9 + [11.0]})
    s = summarize(df, near_failure=5)
    assert np.isclose(s["pooled_mae"], 2.0) and np.isclose(s["unit_avg_mae"], 6.0)
    assert s["overestimate_fraction"] == 1.0
    ci = unit_bootstrap_ci(df, 200, 0)
    assert ci["low"] <= 6.0 <= ci["high"]
