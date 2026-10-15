"""Configuration loading and the dataset gate."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "config.yaml"


class DatasetGateBlocked(RuntimeError):
    """Raised when a dataset-dependent step runs before a dataset is approved."""


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    with open(path or DEFAULT_CONFIG, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(cfg: dict[str, Any], key: str) -> Path:
    """Resolve a configured path relative to the project root."""
    p = Path(cfg["paths"][key])
    return p if p.is_absolute() else PROJECT_ROOT / p


def gate_status(cfg: dict[str, Any]) -> str:
    return str(cfg.get("dataset", {}).get("gate_status", "blocked"))


def require_dataset(cfg: dict[str, Any]) -> dict[str, Any]:
    """Return the dataset spec, or raise DatasetGateBlocked with the exact reason."""
    ds = cfg.get("dataset", {})
    if ds.get("gate_status") != "approved":
        raise DatasetGateBlocked(
            "BLOCKED: no dataset satisfies both the turbofan topic and the "
            "real-measurement-only requirement, and no faculty decision (A or B) "
            "has been recorded in configs/config.yaml. See FEASIBILITY.md."
        )
    missing = [k for k in ("decision", "approved_by", "name", "spec_path") if not ds.get(k)]
    if missing:
        raise DatasetGateBlocked(f"BLOCKED: dataset approved but missing fields: {missing}")
    spec_path = Path(ds["spec_path"])
    if not spec_path.is_absolute():
        spec_path = PROJECT_ROOT / spec_path
    with open(spec_path, encoding="utf-8") as f:
        return yaml.safe_load(f)
