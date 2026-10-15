"""Train baselines and the LSTM on the saved split; save artifacts and predictions."""

import argparse

from _common import gated
from aerolife.config import resolve
from aerolife.pipeline import load_canonical, run_experiment
from aerolife.splits import load_manifest

p = argparse.ArgumentParser()
p.add_argument("--config")
a = p.parse_args()
cfg, spec = gated(a.config)
obs, units = load_canonical(resolve(cfg, "canonical_dir"))
split = load_manifest(resolve(cfg, "splits_dir") / "split_manifest.json")["split"]
out = resolve(cfg, "artifacts_dir") / f"v{cfg['project']['model_version']}"
res = run_experiment(cfg, spec, obs, units, split, out)
print("saved bundle:", out)
for r in res["runs"]:
    print(f"seed {r['seed']}: best val loss {r['best_val_loss']:.4f}, epochs {r['epochs_run']}, "
          f"{r['train_seconds']:.1f}s, params {r['params']}")
