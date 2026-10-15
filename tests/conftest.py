"""Test fixtures.

These are tiny, hand-written, deterministic tables used ONLY to check index
logic (unit boundaries, alignment, causality, serialisation). Channel values
are simple functions of the row index with no physical meaning, there is no
random noise, and the lifetimes are arbitrary integers. They are created in
pytest temporary directories, never written to data/ or artifacts/, never used
to train a reported model, and never shown as results.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

LIVES = {"u1": 40, "u2": 50, "u3": 60, "u4": 45, "u5": 55, "u6": 35, "u7": 48, "u8": 52}


def make_tables(lives=LIVES, censored=("u9",)):
    rows, urows = [], []
    for k, (uid, n) in enumerate(lives.items()):
        for i in range(n):
            rows.append({"unit_id": uid, "time_index": i, "time_value": float(i),
                         "ch_a": float(k * 1000 + i), "ch_b": float(i % 7), "ch_const": 1.0})
        urows.append({"unit_id": uid, "event_observed": True, "failure_time": float(n),
                      "condition_id": "c1" if k % 2 == 0 else "c2", "include": True, "exclusion_reason": ""})
    for uid in censored:
        for i in range(30):
            rows.append({"unit_id": uid, "time_index": i, "time_value": float(i),
                         "ch_a": float(9000 + i), "ch_b": float(i % 7), "ch_const": 1.0})
        urows.append({"unit_id": uid, "event_observed": False, "failure_time": np.nan,
                      "condition_id": "c1", "include": True, "exclusion_reason": ""})
    return pd.DataFrame(rows), pd.DataFrame(urows)


@pytest.fixture
def tables():
    return make_tables()


SPEC = {"name": "TEST FIXTURE (not data)", "time_unit": "step", "time_step_meaning": "fixture row",
        "input_columns": ["ch_a", "ch_b", "ch_const"]}


def small_cfg():
    from aerolife.config import load_config

    cfg = load_config()
    cfg["model"].update(epochs=2, patience=1, seeds=[1], units=4, dense_units=4, batch_size=32)
    cfg["baselines"].update(rf_n_estimators=5)
    cfg["windowing"].update(sequence_length=5, stride=1)
    cfg["evaluation"].update(bootstrap_samples=50, near_failure_threshold=10)
    return cfg
