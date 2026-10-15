"""Compute test tables and figures from saved predictions (no retraining)."""

import argparse

from _common import gated
from aerolife.config import resolve
from aerolife.inference import current_bundle_dir
from aerolife.reporting import write_report

p = argparse.ArgumentParser()
p.add_argument("--config")
a = p.parse_args()
cfg, _ = gated(a.config)
bundle = current_bundle_dir(resolve(cfg, "artifacts_dir"))
if bundle is None:
    raise SystemExit("BLOCKED: no trained artifacts; run scripts/train.py first")
tab = write_report(bundle, resolve(cfg, "reports_dir"), cfg)
print(tab.to_string(index=False))
