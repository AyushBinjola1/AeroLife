"""Sequence-length ablation. Selection uses validation units only; test is reported
for every variant only after selection, never used to choose."""

import argparse

import pandas as pd

from _common import gated
from aerolife.config import resolve
from aerolife.metrics import summarize
from aerolife.pipeline import load_canonical, run_experiment
from aerolife.splits import load_manifest

p = argparse.ArgumentParser()
p.add_argument("--config")
p.add_argument("--lengths", type=int, nargs="+", required=True)
a = p.parse_args()
cfg, spec = gated(a.config)
obs, units = load_canonical(resolve(cfg, "canonical_dir"))
split = load_manifest(resolve(cfg, "splits_dir") / "split_manifest.json")["split"]
seed = cfg["model"]["seeds"][0]
rows = []
for L in a.lengths:
    out = resolve(cfg, "artifacts_dir") / "ablation" / f"L{L}"
    res = run_experiment(cfg, spec, obs, units, split, out, seq_len=L, seeds=[seed], save=False)
    name = f"{cfg['model']['type']}_seed{seed}"
    v = summarize(res["pred_val"][res["pred_val"]["model"] == name])
    t = summarize(res["pred_test"][res["pred_test"]["model"] == name])
    rows.append({"sequence_length": L, "val_unit_avg_mae": v["unit_avg_mae"], "test_unit_avg_mae": t["unit_avg_mae"]})
tab = pd.DataFrame(rows)
tab["selected_by_validation"] = tab["val_unit_avg_mae"] == tab["val_unit_avg_mae"].min()
dest = resolve(cfg, "reports_dir") / "tables" / "ablation_sequence_length.csv"
dest.parent.mkdir(parents=True, exist_ok=True)
tab.to_csv(dest, index=False)
print(tab.to_string(index=False))
