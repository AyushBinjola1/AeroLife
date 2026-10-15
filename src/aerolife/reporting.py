"""Tables and static figures from saved predictions (no retraining)."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .metrics import per_unit_table, phm08_score, summarize, unit_bootstrap_ci  # noqa: E402
from .schema import ID, TVAL  # noqa: E402


def comparison_table(pred: pd.DataFrame, near_failure, n_boot: int, seed: int, cycles: bool) -> pd.DataFrame:
    rows = []
    for name, g in pred.groupby("model", sort=False):
        r = {"model": name, **summarize(g, "rul", "pred", near_failure)}
        ci = unit_bootstrap_ci(g, n_boot, seed)
        r["unit_avg_mae_ci95_low"], r["unit_avg_mae_ci95_high"] = ci["low"], ci["high"]
        r["capped_target_pooled_mae"] = float(np.mean(np.abs(g["pred"] - g["rul_target"])))
        if cycles:
            r["phm08_score"] = phm08_score(g["rul"], g["pred"])
        rows.append(r)
    return pd.DataFrame(rows)


def write_report(run_dir: Path, out_dir: Path, cfg: dict) -> pd.DataFrame:
    run_dir, out_dir = Path(run_dir), Path(out_dir)
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    meta = json.loads((run_dir / "metadata.json").read_text())
    runs = json.loads((run_dir / "training_runs.json").read_text())
    pred = pd.read_csv(run_dir / "predictions_test.csv", dtype={ID: str})
    unit = meta["time_unit"]
    ev = cfg["evaluation"]
    tab = comparison_table(pred, ev.get("near_failure_threshold"), ev["bootstrap_samples"],
                           cfg["split"]["seed"], unit.lower().startswith("cycle"))
    tab.to_csv(out_dir / "tables" / "test_comparison.csv", index=False)

    rnn = pred[pred["model"] == f"{meta['cell']}_seed{meta['selected_seed']}"]
    per_unit_table(rnn).to_csv(out_dir / "tables" / "test_per_unit_selected_rnn.csv", index=False)
    pd.DataFrame(runs).to_csv(out_dir / "tables" / "training_runs.csv", index=False)

    # Predicted vs actual
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(rnn["rul"], rnn["pred"], s=6, alpha=0.5)
    lim = float(max(rnn["rul"].max(), rnn["pred"].max()))
    ax.plot([0, lim], [0, lim], "k--", lw=1)
    ax.set(xlabel=f"actual RUL ({unit})", ylabel=f"predicted RUL ({unit})", title="Held-out test windows")
    fig.tight_layout(); fig.savefig(out_dir / "figures" / "pred_vs_actual.png", dpi=150); plt.close(fig)

    # Trajectories per test unit
    ids = sorted(rnn[ID].unique())
    fig, axes = plt.subplots(len(ids), 1, figsize=(7, 2.2 * len(ids)), squeeze=False)
    for ax, uid in zip(axes[:, 0], ids):
        g = rnn[rnn[ID] == uid].sort_values(TVAL)
        ax.plot(g[TVAL], g["rul"], "k-", label="actual (retrospective)")
        ax.plot(g[TVAL], g["pred"], "-", label="predicted")
        ax.set(title=f"unit {uid}", xlabel=f"elapsed time ({unit})", ylabel=f"RUL ({unit})")
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out_dir / "figures" / "test_trajectories.png", dpi=150); plt.close(fig)

    # Residuals
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(rnn["rul"], rnn["pred"] - rnn["rul"], s=6, alpha=0.5)
    ax.axhline(0, color="k", lw=1)
    ax.set(xlabel=f"actual RUL ({unit})", ylabel=f"pred - actual ({unit})", title="Residuals (>0 = overestimate)")
    fig.tight_layout(); fig.savefig(out_dir / "figures" / "residuals.png", dpi=150); plt.close(fig)

    # Loss curves
    fig, ax = plt.subplots(figsize=(6, 4))
    for r in runs:
        ax.plot(r["history"]["loss"], label=f"train s{r['seed']}")
        ax.plot(r["history"]["val_loss"], "--", label=f"val s{r['seed']}")
    ax.set(xlabel="epoch", ylabel="loss (scaled target)", yscale="log", title="Training / validation loss")
    ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out_dir / "figures" / "loss_curves.png", dpi=150); plt.close(fig)

    # Baseline comparison
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(tab["model"], tab["unit_avg_mae"],
           yerr=[tab["unit_avg_mae"] - tab["unit_avg_mae_ci95_low"], tab["unit_avg_mae_ci95_high"] - tab["unit_avg_mae"]])
    ax.set(ylabel=f"unit-averaged MAE ({unit})", title="Test units: model comparison (95% unit bootstrap)")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout(); fig.savefig(out_dir / "figures" / "baseline_comparison.png", dpi=150); plt.close(fig)
    return tab
