"""End-to-end experiment: unit split -> causal preprocessing -> windows -> models.

Order matters and is enforced here:
1. units are already partitioned (split manifest) before this runs;
2. the preprocessor is fitted on training-unit rows only;
3. windows are built per partition and per unit;
4. LSTM early stopping / LR schedule / checkpointing watch validation units only;
5. test units are touched once, for final prediction.
"""

from __future__ import annotations

import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import baselines as bl
from . import models
from .inference import save_bundle
from .preprocessing import CausalPreprocessor
from .schema import ID, TIDX, TVAL, input_columns
from .targets import add_rul
from .windowing import make_windows, summarize_windows


def load_canonical(canonical_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    obs = pd.read_csv(Path(canonical_dir) / "observations.csv", dtype={ID: str})
    units = pd.read_csv(Path(canonical_dir) / "units.csv", dtype={ID: str})
    return obs, units


def build_partitions(cfg, spec, obs, units, split, seq_len: int | None = None):
    cap = cfg["targets"].get("rul_cap")
    labelled = add_rul(obs, units, cap)
    channels = input_columns(obs, list(spec["input_columns"]))
    part_obs = {k: labelled[labelled[ID].isin(v)] for k, v in split.items()}
    pp = cfg["preprocessing"]
    prep = CausalPreprocessor(channels, pp.get("rolling_window", 0),
                              pp.get("constant_std_threshold", 1e-8),
                              pp.get("include_elapsed_time", True)).fit(part_obs["train"])
    L = int(seq_len or cfg["windowing"]["sequence_length"])
    stride = int(cfg["windowing"]["stride"])
    parts = {}
    for k, o in part_obs.items():
        z = prep.transform(o).merge(o[[ID, TIDX, "rul", "rul_target"]], on=[ID, TIDX])
        X, y, meta = make_windows(z, prep.feature_names, L, stride, target_col="rul_target",
                                  extra_meta=("rul", "rul_target"))
        parts[k] = {"X": X, "y": y, "meta": meta}
    lifetimes = units.set_index(ID).loc[split["train"], "failure_time"].to_numpy(float)
    return prep, parts, lifetimes, L


def run_experiment(cfg, spec, obs, units, split, out_dir: Path, seq_len: int | None = None,
                   seeds: list[int] | None = None, save: bool = True) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prep, parts, lifetimes, L = build_partitions(cfg, spec, obs, units, split, seq_len)
    tr, va, te = parts["train"], parts["val"], parts["test"]
    for k, p in parts.items():
        if len(p["X"]) == 0:
            raise ValueError(f"no {k} windows: sequence_length {L} exceeds every {k} history")
    cap = cfg["targets"].get("rul_cap")
    target_scale = float(cap or tr["y"].max() or 1.0)
    mcfg, bcfg = cfg["model"], cfg["baselines"]
    seeds = seeds or list(mcfg["seeds"])

    preds = {"val": [], "test": []}

    def record(name, split_name, yhat):
        m = parts[split_name]["meta"].copy()
        m["model"], m["pred"] = name, np.maximum(np.asarray(yhat, float), 0.0)
        preds[split_name].append(m)

    # ---- baselines (same windows, same targets, same evaluation times) ----
    const = bl.ConstantBaseline().fit(tr["y"])
    age = bl.AgeBaseline().fit(lifetimes, cap)
    S_tr, _ = summarize_windows(tr["X"], prep.feature_names)
    ridge = bl.make_ridge(bcfg["ridge_alpha"]).fit(S_tr, tr["y"])
    rf = bl.make_rf(bcfg["rf_n_estimators"], bcfg["rf_max_depth"], seeds[0]).fit(S_tr, tr["y"])
    for s in ("val", "test"):
        p = parts[s]
        S, _ = summarize_windows(p["X"], prep.feature_names)
        record(const.name, s, const.predict(len(p["X"])))
        record(age.name, s, age.predict(p["meta"][TVAL].to_numpy()))
        record("ridge", s, ridge.predict(S))
        record("random_forest", s, rf.predict(S))

    # ---- recurrent model, several seeds ----
    runs = []
    for seed in seeds:
        models.set_seeds(seed)
        model = models.build_rnn(L, len(prep.feature_names), mcfg["type"], mcfg["units"],
                                 mcfg["dropout"], mcfg["dense_units"], mcfg["loss"], mcfg["learning_rate"])
        ckpt = out_dir / f"ckpt_seed{seed}.keras"
        hist, secs = models.train(model, tr["X"], tr["y"] / target_scale, va["X"], va["y"] / target_scale,
                                  epochs=mcfg["epochs"], batch_size=mcfg["batch_size"],
                                  patience=mcfg["patience"], checkpoint_path=str(ckpt))
        name = f"{mcfg['type']}_seed{seed}"
        for s in ("val", "test"):
            record(name, s, models.predict(model, parts[s]["X"]) * target_scale)
        t0 = time.perf_counter()
        for _ in range(20):
            model.predict(te["X"][:1], verbose=0)
        latency_ms = (time.perf_counter() - t0) / 20 * 1000
        runs.append({"seed": seed, "model": model, "history": hist, "train_seconds": secs,
                     "best_val_loss": float(min(hist["val_loss"])), "epochs_run": len(hist["loss"]),
                     "latency_ms_single_window": latency_ms, "params": int(model.count_params())})
        ckpt.unlink(missing_ok=True)

    best = min(runs, key=lambda r: r["best_val_loss"])  # chosen on validation only
    pred_val = pd.concat(preds["val"], ignore_index=True)
    pred_test = pd.concat(preds["test"], ignore_index=True)
    pred_val.to_csv(out_dir / "predictions_val.csv", index=False)
    pred_test.to_csv(out_dir / "predictions_test.csv", index=False)
    run_info = [{k: v for k, v in r.items() if k != "model"} for r in runs]
    (out_dir / "training_runs.json").write_text(json.dumps(run_info, indent=2), encoding="utf-8")

    import tensorflow as tf

    meta = {
        "dataset_name": spec["name"], "time_unit": spec["time_unit"],
        "time_step_meaning": spec.get("time_step_meaning"),
        "model_version": cfg["project"]["model_version"], "cell": mcfg["type"],
        "selected_seed": best["seed"], "sequence_length": L, "stride": cfg["windowing"]["stride"],
        "feature_names": prep.feature_names, "channels": prep.channels,
        "target_scale": target_scale, "rul_cap": cap,
        "split": split, "parameter_count": best["params"],
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "tensorflow": tf.__version__, "python": platform.python_version(),
    }
    if save:
        save_bundle(out_dir, best["model"], prep, meta)
    return {"meta": meta, "runs": run_info, "pred_val": pred_val, "pred_test": pred_test}
