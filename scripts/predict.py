"""Predict RUL for a history CSV with the saved model + preprocessing artifacts."""

import argparse
import sys

import pandas as pd

from _common import ROOT  # noqa: F401
from aerolife.config import load_config, resolve
from aerolife.inference import current_bundle_dir, load_bundle, predict_history
from aerolife.schema import ID

p = argparse.ArgumentParser()
p.add_argument("history_csv")
p.add_argument("--out", default="predictions.csv")
p.add_argument("--config")
a = p.parse_args()
cfg = load_config(a.config)
bdir = current_bundle_dir(resolve(cfg, "artifacts_dir"))
if bdir is None:
    sys.exit("BLOCKED: no trained artifacts exist; the model has not been trained.")
bundle = load_bundle(bdir)
df = pd.read_csv(a.history_csv, dtype={ID: str})
pred, rep = predict_history(df, bundle)
for w in rep.warnings:
    print("warning:", w)
if not rep.ok:
    sys.exit("schema validation failed:\n" + "\n".join(rep.errors))
pred.to_csv(a.out, index=False)
print(f"wrote {len(pred)} rows to {a.out} (time unit: {bundle.meta['time_unit']})")
