import numpy as np
import pandas as pd
import pytest

from aerolife.adapters import REGISTRY
from aerolife.pipeline import build_partitions
from aerolife.targets import add_rul
from conftest import SPEC, small_cfg


def test_official_label_alignment_and_distinct_ids(tmp_path):
    folder = tmp_path / "CMAPSSData"
    folder.mkdir()
    rows = [[uid, cycle, *([float(uid)] * 24)] for uid in range(1, 101) for cycle in (1, 2)]
    for part in ("train", "test"):
        pd.DataFrame(rows).to_csv(folder / f"{part}_FD001.txt", sep=" ", header=False, index=False)
    pd.Series(range(1, 101)).to_csv(folder / "RUL_FD001.txt", header=False, index=False)
    obs, units = REGISTRY["cmapss_fd001"]().parse(tmp_path)
    labelled = add_rul(obs, units)
    final = labelled.groupby("unit_id").tail(1).set_index("unit_id")
    assert len(units) == 200
    assert final.loc["train_001", "rul"] == 0
    assert final.loc["test_001", "rul"] == 1
    assert final.loc["test_100", "rul"] == 100
    assert "rul" not in obs.columns
    (folder / "RUL_FD001.txt").write_text("1\n")
    with pytest.raises(ValueError):
        REGISTRY["cmapss_fd001"]().parse(tmp_path)


def test_official_endpoint_selection(tables):
    obs, units = tables
    split = dict(train=["u1", "u2", "u3", "u4"], val=["u5", "u6"], test=["u7", "u8"])
    cfg = small_cfg()
    _, all_parts, _, _ = build_partitions(cfg, SPEC, obs, units, split)
    _, parts, _, _ = build_partitions(cfg, {**SPEC, "test_last_window_only": True}, obs, units, split)
    expected = all_parts["test"]["meta"].groupby("unit_id").tail(1)
    pd.testing.assert_frame_equal(parts["test"]["meta"], expected.reset_index(drop=True))
    np.testing.assert_array_equal(parts["test"]["X"], all_parts["test"]["X"][expected.index])
    assert len(parts["test"]["X"]) == 2
    assert len(parts["train"]["X"]) == len(all_parts["train"]["X"])


@pytest.mark.parametrize("invalid", ["unknown_engine", "u9"])
def test_partition_rejects_unknown_and_censored_units(tables, invalid):
    obs, units = tables
    split = dict(train=["u1", "u2"], val=["u3"], test=[invalid])
    with pytest.raises(ValueError, match="unknown, excluded or censored"):
        build_partitions(small_cfg(), SPEC, obs, units, split)


def test_partition_rejects_overlap_before_preprocessing(tables):
    obs, units = tables
    split = dict(train=["u1", "u2"], val=["u2"], test=["u3"])
    with pytest.raises(AssertionError, match="both"):
        build_partitions(small_cfg(), SPEC, obs, units, split)


def test_official_evaluation_refuses_to_silently_drop_short_engines(tables):
    obs, units = tables
    obs = obs[(obs.unit_id != "u8") | (obs.time_index < 2)]
    split = dict(train=["u1", "u2"], val=["u3"], test=["u7", "u8"])
    with pytest.raises(ValueError, match="every test engine"):
        build_partitions(small_cfg(), {**SPEC, "test_last_window_only": True}, obs, units, split)



def test_official_evaluation_refuses_stale_endpoint_after_gap(tables):
    obs, units = tables
    obs = obs[~((obs.unit_id == "u8") & (obs.time_index == 50))]
    split = dict(train=["u1", "u2"], val=["u3"], test=["u7", "u8"])
    with pytest.raises(ValueError, match="final observation"):
        build_partitions(small_cfg(), {**SPEC, "test_last_window_only": True}, obs, units, split)
