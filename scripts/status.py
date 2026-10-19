"""Print what is complete and what is blocked."""

import argparse

from _common import ROOT  # noqa: F401
from aerolife.config import gate_status, load_config, resolve
from aerolife.inference import current_bundle_dir

p = argparse.ArgumentParser()
p.add_argument("--config")
a = p.parse_args()
cfg = load_config(a.config)
ds = cfg["dataset"]
print(f"dataset gate      : {gate_status(cfg)}")
print(f"dataset decision  : {ds.get('decision') or 'pending (A or B, see FEASIBILITY.md)'}")
canon = resolve(cfg, "canonical_dir") / "observations.csv"
print(f"canonical data    : {'present' if canon.exists() else 'absent'}")
split = resolve(cfg, "splits_dir") / "split_manifest.json"
print(f"split manifest    : {'present' if split.exists() else 'absent'}")
b = current_bundle_dir(resolve(cfg, "artifacts_dir"))
print(f"trained artifacts : {b if b else 'absent (model NOT trained)'}")
