"""Saved-artifact inference. Ground truth is never an input here."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .preprocessing import CausalPreprocessor
from .schema import ID, TIDX, TVAL, ValidationReport, validate_observations
from .windowing import make_windows

INSUFFICIENT = "insufficient history"


@dataclass
class Bundle:
    path: Path
    model: object
    prep: CausalPreprocessor
    meta: dict

    @property
    def seq_len(self) -> int:
        return int(self.meta["sequence_length"])


def current_bundle_dir(artifacts_dir: Path) -> Path | None:
    pointer = Path(artifacts_dir) / "CURRENT"
    if not pointer.exists():
        return None
    d = Path(artifacts_dir) / pointer.read_text(encoding="utf-8").strip()
    return d if (d / "model.keras").exists() and (d / "metadata.json").exists() else None


def save_bundle(out_dir: Path, model, prep: CausalPreprocessor, meta: dict, make_current: bool = True) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save(out_dir / "model.keras")
    (out_dir / "preprocessor.json").write_text(json.dumps(prep.to_dict(), indent=2), encoding="utf-8")
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    if make_current:
        (out_dir.parent / "CURRENT").write_text(out_dir.name, encoding="utf-8")
    return out_dir


def load_bundle(bundle_dir: Path) -> Bundle:
    import keras

    bundle_dir = Path(bundle_dir)
    model = keras.models.load_model(bundle_dir / "model.keras", compile=False)
    prep = CausalPreprocessor.from_dict(json.loads((bundle_dir / "preprocessor.json").read_text(encoding="utf-8")))
    meta = json.loads((bundle_dir / "metadata.json").read_text(encoding="utf-8"))
    return Bundle(bundle_dir, model, prep, meta)


def predict_history(obs: pd.DataFrame, bundle: Bundle) -> tuple[pd.DataFrame, ValidationReport]:
    """Predict RUL at every observed step of each unit, using only data up to that step.

    Returns one row per observation: predicted_rul is NaN with status
    'insufficient history' until sequence_length observations exist.
    """
    rep = validate_observations(obs, channels=bundle.prep.channels)
    if not rep.ok:
        return pd.DataFrame(), rep
    clean = obs.drop(columns=rep.dropped_columns)[[ID, TIDX, TVAL, *bundle.prep.channels]].copy()
    clean[ID] = clean[ID].astype(str)
    z = bundle.prep.transform(clean)
    X, _, meta = make_windows(z, bundle.prep.feature_names, bundle.seq_len, stride=1, extra_meta=())
    out = z[[ID, TIDX, TVAL]].copy()
    out["status"] = INSUFFICIENT
    out["predicted_rul"] = np.nan
    if len(X):
        from .models import predict

        y = predict(bundle.model, X) * float(bundle.meta["target_scale"])
        key = pd.MultiIndex.from_frame(meta[[ID, TIDX]])
        pos = pd.MultiIndex.from_frame(out[[ID, TIDX]]).get_indexer(key)
        out.loc[out.index[pos], "predicted_rul"] = y
        out.loc[out.index[pos], "status"] = "ok"
    out["time_unit"] = bundle.meta.get("time_unit")
    out["dataset"] = bundle.meta.get("dataset_name")
    out["model_version"] = bundle.meta.get("model_version")
    return out, rep
