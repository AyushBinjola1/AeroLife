"""Shared CLI helpers: put src/ on the path and enforce the dataset gate."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aerolife.config import DatasetGateBlocked, load_config, require_dataset  # noqa: E402


def gated(cfg_path=None):
    cfg = load_config(cfg_path)
    try:
        spec = require_dataset(cfg)
    except DatasetGateBlocked as e:
        print(e, file=sys.stderr)
        sys.exit(2)
    return cfg, spec
