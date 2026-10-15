import json

import numpy as np
import pytest

from aerolife.splits import (assert_disjoint, explicit_split, group_split, leave_one_unit_out,
                             save_manifest)
from aerolife.windowing import make_windows, summarize_windows, window_ends


def test_group_split_disjoint_complete_and_excludes_censored(tables, tmp_path):
    _, units = tables
    s = group_split(units, 0.25, 0.25, seed=0, stratify_col="condition_id")
    assert_disjoint(s)
    assigned = set(s["train"]) | set(s["val"]) | set(s["test"])
    assert assigned == {f"u{i}" for i in range(1, 9)}
    assert "u9" not in assigned
    save_manifest(s, units, tmp_path / "m.json", {"dataset": "fixture"})
    m = json.loads((tmp_path / "m.json").read_text())
    assert m["censored_units"] == ["u9"] and set(m["conditions_per_partition"]["test"]) <= {"c1", "c2"}


def test_split_is_reproducible(tables):
    _, units = tables
    assert group_split(units, 0.25, 0.25, 3) == group_split(units, 0.25, 0.25, 3)


def test_overlap_rejected():
    with pytest.raises(AssertionError):
        explicit_split(["u1", "u2"], ["u2"], ["u3"])


def test_louo():
    folds = list(leave_one_unit_out(["a", "b", "c"]))
    assert len(folds) == 3 and all(v not in tr for tr, v in folds)


def test_window_ends_anchor_last_observation():
    np.testing.assert_array_equal(window_ends(10, 3, 4), [5, 9])  # e=1 would need index -1
    np.testing.assert_array_equal(window_ends(10, 3, 1), np.arange(2, 10))
    assert window_ends(2, 3, 1).size == 0


def test_windows_never_cross_units_and_label_last_step(tables):
    obs, _ = tables
    obs = obs.assign(rul=obs.time_value * -1)  # marker column for alignment
    L = 6
    X, y, meta = make_windows(obs, ["ch_a", "time_index"], L, stride=2, target_col="rul")
    unit_code = X[:, :, 0] // 1000
    assert (unit_code == unit_code[:, :1]).all()                      # single unit per window
    assert (np.diff(X[:, :, 1], axis=1) == 1).all()                    # consecutive steps
    np.testing.assert_array_equal(X[:, -1, 1], meta.time_index)        # label at last step
    np.testing.assert_allclose(y, -meta.time_value)
    per_unit = meta.groupby("unit_id").time_index.max()
    lens = obs.groupby("unit_id").time_index.max()
    assert (per_unit == lens.loc[per_unit.index]).all()                # final step always included


def test_short_history_yields_no_windows(tables):
    obs, _ = tables
    X, _, meta = make_windows(obs[obs.unit_id == "u1"], ["ch_a"], 41)
    assert len(X) == 0 and meta.empty


def test_summary_features_shape():
    X = np.arange(2 * 4 * 3, dtype=np.float32).reshape(2, 4, 3)
    S, names = summarize_windows(X, ["a", "b", "c"])
    assert S.shape == (2, 18) and len(names) == 18
    np.testing.assert_allclose(S[:, names.index("slope_a")], 3.0)  # +3 per step in this array
