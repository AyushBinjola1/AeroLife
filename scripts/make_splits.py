"""Partition whole units into train/val/test BEFORE windowing; save the manifest."""

import argparse

from _common import gated
from aerolife.config import resolve
from aerolife.pipeline import load_canonical
from aerolife.splits import explicit_split, group_split, save_manifest

p = argparse.ArgumentParser()
p.add_argument("--config")
a = p.parse_args()
cfg, spec = gated(a.config)
obs, units = load_canonical(resolve(cfg, "canonical_dir"))
s = cfg["split"]
if s.get("train_units"):
    split = explicit_split(s["train_units"], s["val_units"] or [], s["test_units"] or [])
else:
    strat = "condition_id" if s.get("stratify_by_condition") else None
    split = group_split(units, s["val_fraction"], s["test_fraction"], s["seed"], strat)
save_manifest(split, units, resolve(cfg, "splits_dir") / "split_manifest.json",
              {"dataset": spec["name"], "method": s["method"], "seed": s["seed"]})
print({k: len(v) for k, v in split.items()})
