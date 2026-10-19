"""Unit-level splits, created before any windowing."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .schema import ID


def group_split(
    units: pd.DataFrame,
    val_fraction: float,
    test_fraction: float,
    seed: int,
    stratify_col: str | None = None,
) -> dict[str, list[str]]:
    """Assign whole units to train/val/test. Stratifies by condition when asked,
    so each condition is represented where possible (interpolation test)."""
    if not (0 <= val_fraction < 1 and 0 <= test_fraction < 1 and val_fraction + test_fraction < 1):
        raise ValueError("split fractions must be non-negative and sum to less than one")
    eligible = units[units["include"].astype(bool) & units["event_observed"].astype(bool)]
    rng = np.random.default_rng(seed)
    groups = (
        [g for _, g in eligible.groupby(stratify_col, sort=True)]
        if stratify_col and stratify_col in eligible.columns
        else [eligible]
    )
    split: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for g in groups:
        ids = sorted(g[ID].astype(str))
        rng.shuffle(ids)
        n = len(ids)
        n_test = int(round(n * test_fraction))
        n_val = int(round(n * val_fraction))
        if n >= 3:
            n_test = max(n_test, 1) if test_fraction else 0
            n_val = max(n_val, 1) if val_fraction else 0
            while n_test + n_val >= n:
                if n_test >= n_val:
                    n_test -= 1
                else:
                    n_val -= 1
        split["test"] += ids[:n_test]
        split["val"] += ids[n_test : n_test + n_val]
        split["train"] += ids[n_test + n_val :]
    for k in split:
        split[k] = sorted(split[k])
    assert_disjoint(split)
    if not split["train"]:
        raise ValueError("no training units after split")
    return split


def explicit_split(train: list, val: list, test: list) -> dict[str, list[str]]:
    split = {"train": sorted(map(str, train)), "val": sorted(map(str, val)), "test": sorted(map(str, test))}
    assert_disjoint(split)
    return split


def leave_one_unit_out(unit_ids: list[str]):
    """Yield (train_ids, val_id) folds over the given (non-test) units."""
    ids = sorted(map(str, unit_ids))
    for held in ids:
        yield [u for u in ids if u != held], held


def assert_disjoint(split: dict[str, list[str]]) -> None:
    names = list(split)
    for name, ids in split.items():
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate units in {name} partition")
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            both = set(split[a]) & set(split[b])
            if both:
                raise AssertionError(f"units appear in both {a} and {b}: {sorted(both)}")


def save_manifest(split: dict[str, list[str]], units: pd.DataFrame, path: Path, meta: dict) -> None:
    cond = "condition_id" if "condition_id" in units.columns else None
    conditions = {}
    if cond:
        lookup = units.set_index(units[ID].astype(str))[cond].astype(str)
        conditions = {k: sorted(set(lookup.loc[v])) if v else [] for k, v in split.items()}
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "split": split,
        "counts": {k: len(v) for k, v in split.items()},
        "conditions_per_partition": conditions,
        "excluded_units": units.loc[~units["include"].astype(bool), [ID, "exclusion_reason"]]
        .astype(str).to_dict("records"),
        "censored_units": units.loc[~units["event_observed"].astype(bool), ID].astype(str).tolist(),
        **meta,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def load_manifest(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
