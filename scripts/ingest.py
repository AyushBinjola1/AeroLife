"""Parse immutable raw files with a registered adapter, validate, write canonical tables
and the provenance manifest."""

import argparse
import json
import sys

from _common import gated
from aerolife.adapters import REGISTRY
from aerolife.config import resolve
from aerolife.provenance import build_manifest, verify_manifest, write_manifest
from aerolife.schema import validate_observations, validate_units

p = argparse.ArgumentParser()
p.add_argument("--config")
a = p.parse_args()
cfg, spec = gated(a.config)
adapter_name = spec.get("adapter")
if adapter_name not in REGISTRY:
    sys.exit(f"BLOCKED: no adapter registered for '{adapter_name}'. See src/aerolife/adapters/.")

raw = resolve(cfg, "raw_dir")
manifest_path = raw.parent / "provenance_manifest.json"
if manifest_path.exists():
    problems = verify_manifest(raw, manifest_path)
    if problems:
        sys.exit("raw data changed since manifest was written:\n" + "\n".join(problems))
else:
    write_manifest(build_manifest(raw, spec["source"]), manifest_path)

obs, units = REGISTRY[adapter_name]().parse(raw)
r1 = validate_observations(obs, channels=spec["input_columns"])
r2 = validate_units(units, obs)
for r in (r1, r2):
    for w in r.warnings:
        print("warning:", w)
if not (r1.ok and r2.ok):
    sys.exit("schema validation failed:\n" + "\n".join(r1.errors + r2.errors))
out = resolve(cfg, "canonical_dir")
out.mkdir(parents=True, exist_ok=True)
obs.to_csv(out / "observations.csv", index=False)
units.to_csv(out / "units.csv", index=False)
counts = {"units_total": int(len(units)), "units_included": int(units["include"].sum()),
          "units_censored": int((~units["event_observed"].astype(bool)).sum()),
          "observations": int(len(obs))}
(out / "ingest_summary.json").write_text(json.dumps(counts, indent=2))
print(counts)
