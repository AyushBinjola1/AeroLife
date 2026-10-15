import subprocess
import sys

import numpy as np
import pytest

from conftest import ROOT, SPEC, small_cfg
from aerolife.config import DatasetGateBlocked, load_config, require_dataset


def test_default_config_is_blocked():
    with pytest.raises(DatasetGateBlocked):
        require_dataset(load_config())


@pytest.mark.parametrize("script", ["ingest.py", "make_splits.py", "train.py", "evaluate.py"])
def test_dataset_scripts_refuse_while_blocked(script):
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / script)], capture_output=True, text=True)
    assert r.returncode == 2 and "BLOCKED" in r.stderr


def test_end_to_end_pipeline_and_saved_inference(tables, tmp_path):
    """Pipeline mechanics on a logic fixture: split -> fit -> windows -> train -> save -> load -> predict."""
    from aerolife.inference import INSUFFICIENT, current_bundle_dir, load_bundle, predict_history
    from aerolife.models import predict
    from aerolife.pipeline import build_partitions, run_experiment
    from aerolife.splits import explicit_split

    obs, units = tables
    cfg = small_cfg()
    split = explicit_split(["u1", "u2", "u3", "u4"], ["u5", "u6"], ["u7", "u8"])
    res = run_experiment(cfg, SPEC, obs, units, split, tmp_path / "v0")
    assert set(res["pred_test"].unit_id) == {"u7", "u8"}
    assert set(res["pred_test"].model) >= {"constant_median", "age_mean_residual_life", "ridge",
                                           "random_forest", "lstm_seed1"}
    bdir = current_bundle_dir(tmp_path)
    assert bdir is not None
    b = load_bundle(bdir)
    assert b.prep.fitted_on_units == ["u1", "u2", "u3", "u4"]

    # Saved-artifact inference equals the in-pipeline prediction for the same windows.
    _, parts, _, _ = build_partitions(cfg, SPEC, obs, units, split)
    direct = predict(b.model, parts["test"]["X"]) * b.meta["target_scale"]
    u = obs[obs.unit_id.isin(["u7", "u8"])]
    out, rep = predict_history(u, b)
    assert rep.ok
    got = out[out.status == "ok"].sort_values(["unit_id", "time_index"]).predicted_rul.to_numpy()
    np.testing.assert_allclose(got, direct, rtol=1e-5)
    first = out[out.unit_id == "u7"].sort_values("time_index")
    assert (first.status.iloc[:4] == INSUFFICIENT).all() and first.predicted_rul.iloc[:4].isna().all()

    # Label columns in an input file cannot influence predictions.
    leaky = u.assign(rul=999.0, failure_time=-1.0)
    out2, rep2 = predict_history(leaky, b)
    assert "rul" in rep2.dropped_columns
    np.testing.assert_allclose(out2.predicted_rul.to_numpy(), out.predicted_rul.to_numpy())

    # Truncating the history gives identical predictions for the retained steps (causal inference).
    trunc, _ = predict_history(u[u.time_index <= 20], b)
    full = out[out.time_index <= 20].reset_index(drop=True)
    np.testing.assert_allclose(trunc.predicted_rul.to_numpy(), full.predicted_rul.to_numpy(), rtol=1e-5)

    # Wrong schema is rejected.
    _, bad = predict_history(u.drop(columns="ch_b"), b)
    assert not bad.ok
