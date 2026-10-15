"""AeroLife dashboard page implementations (entry point: streamlit_app.py).

Shows only recorded data and outputs of saved artifacts. Without an approved
dataset and a trained bundle every data page shows an empty state; nothing is
generated to fill it.
"""

from __future__ import annotations

import io
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aerolife.config import gate_status, load_config, resolve  # noqa: E402
from aerolife.inference import INSUFFICIENT, current_bundle_dir, load_bundle, predict_history  # noqa: E402
from aerolife.schema import ID, TIDX, TVAL, is_forbidden_input, validate_observations  # noqa: E402

CFG = load_config(os.environ.get("AEROLIFE_CONFIG"))
ARTIFACTS = Path(os.environ.get("AEROLIFE_ARTIFACTS_DIR") or resolve(CFG, "artifacts_dir"))
CANONICAL = Path(os.environ.get("AEROLIFE_CANONICAL_DIR") or resolve(CFG, "canonical_dir"))
REPORTS = Path(os.environ.get("AEROLIFE_REPORTS_DIR") or resolve(CFG, "reports_dir"))
GATE = gate_status(CFG)

NOT_LIVE = "Recorded-data replay of a held-out unit — not a live equipment connection."
DISCLAIMER = ("Assignment context only: this project has no affiliation with Rolls-Royce, uses none of its "
              "data, and has no operational validation.")


# ---------------------------------------------------------------- loaders
@st.cache_resource
def _load_bundle(path: str, mtime: float):
    return load_bundle(Path(path))


def get_bundle():
    d = current_bundle_dir(ARTIFACTS)
    return _load_bundle(str(d), (d / "metadata.json").stat().st_mtime) if d else None


@st.cache_data
def _read_tables(o: str, u: str, mtime: float):
    return pd.read_csv(o, dtype={ID: str}), pd.read_csv(u, dtype={ID: str})


def get_data():
    o, u = CANONICAL / "observations.csv", CANONICAL / "units.csv"
    if not (o.exists() and u.exists()):
        return None, None
    return _read_tables(str(o), str(u), o.stat().st_mtime + u.stat().st_mtime)


def empty_state(what: str) -> None:
    st.info(
        f"**{what} unavailable.** Dataset gate: `{GATE}`. "
        + ("No permitted dataset has been approved (see FEASIBILITY.md), so there is no recorded data "
           "and no trained model. Nothing is simulated to fill this page."
           if GATE != "approved" else
           "The dataset is approved but the required step has not been run yet "
           "(ingest → make_splits → train → evaluate).")
    )


def unit_label(bundle) -> str:
    return bundle.meta.get("time_unit", "?") if bundle else "?"


def partition_of(bundle) -> dict:
    if not bundle:
        return {}
    return {u: k for k, ids in bundle.meta["split"].items() for u in ids}


def provenance_caption(bundle, t=None) -> str:
    s = f"Dataset: **{bundle.meta['dataset_name']}** · model v{bundle.meta['model_version']} " \
        f"({bundle.meta['cell'].upper()}, seed {bundle.meta['selected_seed']}) · time unit: **{unit_label(bundle)}**"
    if t is not None:
        s += f" · observed up to elapsed time **{t:g} {unit_label(bundle)}**"
    return s


@st.cache_data(show_spinner=False)
def _predict_upto(uid: str, t_idx: int, bundle_key: str, data_key: str):
    """Predict from the unit's history truncated at t_idx: future rows never reach the model."""
    bundle = get_bundle()
    obs, _ = get_data()
    hist = obs[(obs[ID] == uid) & (obs[TIDX] <= t_idx)]
    pred, _ = predict_history(hist, bundle)
    return pred


def predict_upto(uid: str, t_idx: int):
    b = current_bundle_dir(ARTIFACTS)
    o = CANONICAL / "observations.csv"
    return _predict_upto(uid, t_idx, f"{b}:{(b / 'metadata.json').stat().st_mtime}", f"{o}:{o.stat().st_mtime}")


# ---------------------------------------------------------------- pages
def page_overview():
    st.title("AeroLife: Remaining Useful Life Estimation from Multi-Sensor Time Series")
    st.caption(DISCLAIMER)
    bundle = get_bundle()
    c1, c2, c3 = st.columns(3)
    c1.metric("Dataset gate", GATE)
    c2.metric("Faculty decision", CFG["dataset"].get("decision") or "pending")
    c3.metric("Trained model", "yes" if bundle else "no")
    if GATE != "approved":
        st.warning(
            "An exact dataset satisfying both the turbofan assignment and the real-measurement-only "
            "requirement has not been verified. Training and evaluation are **blocked** pending a "
            "faculty decision: **A** — keep the aircraft-engine topic with a clearly disclosed "
            "simulation benchmark; **B** — keep real measured data and change to rotating machinery "
            "(e.g. bearings)."
        )
    st.subheader("Dataset feasibility table")
    ft = ROOT / "docs" / "feasibility_table.csv"
    if ft.exists():
        st.dataframe(pd.read_csv(ft), hide_index=True, use_container_width=True)
    card = ROOT / "docs" / "dataset_card.md"
    if card.exists():
        st.subheader("Dataset card")
        st.markdown(card.read_text(encoding="utf-8"))
    man = resolve(CFG, "raw_dir").parent / "provenance_manifest.json"
    if man.exists():
        m = json.loads(man.read_text())
        st.subheader("Raw-file provenance")
        st.json(m["source"])
        st.dataframe(pd.DataFrame(m["files"]), hide_index=True)


def page_fleet():
    st.title("Fleet / Equipment Explorer")
    obs, units = get_data()
    if obs is None:
        return empty_state("Equipment data")
    bundle = get_bundle()
    part = partition_of(bundle)
    life = obs.groupby(ID).agg(observations=(TIDX, "count"), last_observed=(TVAL, "max")).reset_index()
    tab = units.merge(life, on=ID, how="left")
    tab["partition"] = tab[ID].map(part).fillna("not in split")
    st.dataframe(tab, hide_index=True, use_container_width=True)
    st.caption("Failure times come from the publisher's documented end-of-life definition and are shown "
               "for retrospective evaluation only; they are never model inputs. Censored units have no failure time.")


def _select_unit(obs, bundle, key):
    part = partition_of(bundle)
    test = [u for u, p in part.items() if p == "test"]
    choices = sorted(test) if test else sorted(obs[ID].unique())
    uid = st.selectbox("Held-out unit" if test else "Unit", choices, key=key)
    g = obs[obs[ID] == uid].sort_values(TIDX)
    t_idx = st.slider("Recorded-data replay: advance through observed history (time_index)",
                      int(g[TIDX].min()), int(g[TIDX].max()), int(g[TIDX].min()), key=key + "_t")
    return uid, g, t_idx


def page_sensors():
    st.title("Sensor History")
    obs, _ = get_data()
    if obs is None:
        return empty_state("Sensor history")
    st.caption(NOT_LIVE)
    bundle = get_bundle()
    uid, g, t_idx = _select_unit(obs, bundle, "sens")
    seen = g[g[TIDX] <= t_idx]
    chans = [c for c in obs.columns if c not in (ID, TIDX, TVAL) and not is_forbidden_input(c)]
    pick = st.multiselect("Channels", chans, default=chans[: min(4, len(chans))])
    for c in pick:
        fig = go.Figure(go.Scatter(x=seen[TVAL], y=seen[c], mode="lines", name=c))
        fig.update_layout(height=220, margin=dict(l=10, r=10, t=30, b=10), title=c,
                          xaxis_title=f"elapsed time ({unit_label(bundle)})")
        st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Showing {len(seen)} of {len(g)} recorded observations — only data up to the selected time.")


def page_prediction():
    st.title("RUL Prediction")
    bundle = get_bundle()
    obs, units = get_data()
    if bundle is None or obs is None:
        empty_state("Prediction")
    else:
        st.caption(NOT_LIVE)
        uid, g, t_idx = _select_unit(obs, bundle, "pred")
        pred = predict_upto(uid, t_idx)
        now = pred.iloc[-1]
        st.markdown(provenance_caption(bundle, float(now[TVAL])))
        if now["status"] == INSUFFICIENT:
            st.warning(f"Insufficient history: {len(pred)} of {bundle.seq_len} required observations. "
                       "No prediction is made; nothing is padded.")
        else:
            st.metric(f"Predicted RUL ({unit_label(bundle)})", f"{now['predicted_rul']:.1f}")
        fig = go.Figure(go.Scatter(x=pred[TVAL], y=pred["predicted_rul"], name="predicted RUL"))
        with st.expander("Retrospective evaluation (ground truth — not an input to the model)"):
            u = units.set_index(ID).loc[uid]
            if bool(u["event_observed"]):
                truth = float(u["failure_time"]) - g[TVAL]
                fig.add_trace(go.Scatter(x=g[TVAL], y=truth, name="actual RUL (retrospective)",
                                         line=dict(dash="dash")))
                st.write(f"Documented failure time: {u['failure_time']:g} {unit_label(bundle)}")
            else:
                st.write("Censored unit: failure not documented, no true RUL exists.")
        fig.update_layout(height=350, xaxis_title=f"elapsed time ({unit_label(bundle)})",
                          yaxis_title=f"RUL ({unit_label(bundle)})")
        st.plotly_chart(fig, use_container_width=True)
        st.download_button("Download predictions (CSV)", pred.to_csv(index=False), f"predictions_{uid}.csv")
        st.download_button("Download concise analysis (Markdown)", analysis_md(bundle, uid, pred),
                           f"analysis_{uid}.md")

    st.divider()
    st.subheader("Upload a history file")
    st.caption("Required columns: unit_id, time_index (int), time_value (elapsed time in the dataset's unit), "
               "plus the measured channels the model was trained on. Label-like columns (RUL, failure time, "
               "health parameters) are ignored.")
    up = st.file_uploader("CSV", type="csv")
    if up is not None:
        df = pd.read_csv(io.BytesIO(up.getvalue()), dtype={ID: str})
        if bundle is None:
            rep = validate_observations(df)
            show_report(rep)
            st.info("Structure checked, but no trained model exists, so no prediction is made.")
        else:
            pred, rep = predict_history(df, bundle)
            show_report(rep)
            if rep.ok:
                st.markdown(provenance_caption(bundle))
                st.dataframe(pred, hide_index=True)
                st.download_button("Download upload predictions (CSV)", pred.to_csv(index=False),
                                   "upload_predictions.csv")


def show_report(rep):
    for e in rep.errors:
        st.error(e)
    for w in rep.warnings:
        st.warning(w)
    if rep.ok:
        st.success("Schema validation passed.")


def analysis_md(bundle, uid, pred) -> str:
    ok = pred[pred["status"] == "ok"]
    last = pred.iloc[-1]
    lines = [
        f"# AeroLife prediction summary — unit {uid}",
        f"- Generated: {datetime.now(timezone.utc).isoformat()}",
        f"- Dataset: {bundle.meta['dataset_name']}; model v{bundle.meta['model_version']} "
        f"({bundle.meta['cell']}, seq. length {bundle.seq_len})",
        f"- Time unit: {unit_label(bundle)}; observed up to {last[TVAL]:g}",
        f"- Latest status: {last['status']}"
        + (f"; predicted RUL {last['predicted_rul']:.1f} {unit_label(bundle)}" if last["status"] == "ok" else ""),
        f"- Steps with a prediction: {len(ok)} of {len(pred)}",
        "",
        "Source: recorded-data replay of a held-out unit. Not a live connection, not a "
        "maintenance or airworthiness decision.",
    ]
    return "\n".join(lines)


def page_maintenance():
    st.title("Maintenance Planning Demonstration")
    st.error("Demonstration only. Thresholds below are configurable teaching values — not manufacturer-approved, "
             "not certified, and not sufficient for any aircraft dispatch or maintenance decision.")
    bundle = get_bundle()
    obs, _ = get_data()
    if bundle is None or obs is None:
        return empty_state("Maintenance demonstration")
    mcfg = CFG["maintenance_demo"]
    tu = unit_label(bundle)
    warn = st.number_input(f"Plan-maintenance threshold ({tu})", min_value=0.0,
                           value=float(mcfg.get("warn_threshold") or 0.0))
    crit = st.number_input(f"Critical threshold ({tu})", min_value=0.0,
                           value=float(mcfg.get("critical_threshold") or 0.0))
    if warn <= crit:
        st.info("Set the plan threshold above the critical threshold to see alerts.")
        return
    st.caption(NOT_LIVE + " " + provenance_caption(bundle))
    st.markdown("For each held-out unit, the replay finds the first recorded time at which the predicted RUL "
                "fell to each threshold, then shows retrospectively how much documented life actually remained "
                "at that moment (negative = the alert would have come after the documented failure).")
    _, units = get_data()
    ulook = units.set_index(ID)
    rows = []
    for uid in sorted(u for u, p in partition_of(bundle).items() if p == "test"):
        # predict_history is causal: the value at each step uses only rows up to that step.
        pred = predict_upto(uid, int(obs.loc[obs[ID] == uid, TIDX].max()))
        ok = pred[pred["status"] == "ok"]
        row = {ID: uid}
        for name, thr in (("plan", warn), ("critical", crit)):
            hit = ok[ok["predicted_rul"] <= thr]
            t = float(hit[TVAL].iloc[0]) if len(hit) else None
            row[f"first {name} alert at ({tu})"] = t
            if t is not None and bool(ulook.loc[uid, "event_observed"]):
                row[f"actual RUL at {name} alert ({tu}, retrospective)"] = float(ulook.loc[uid, "failure_time"]) - t
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


def page_evaluation():
    st.title("Model Evaluation and Limitations")
    tab = REPORTS / "tables" / "test_comparison.csv"
    if not tab.exists():
        empty_state("Evaluation results")
    else:
        bundle = get_bundle()
        if bundle:
            st.markdown(provenance_caption(bundle))
        st.subheader("Held-out test units: model comparison")
        st.dataframe(pd.read_csv(tab), hide_index=True, use_container_width=True)
        st.caption("unit_avg_* = per-unit metric averaged over units (primary; units are the independent samples). "
                   "pooled_* = every window weighted equally, so long-lived units dominate. "
                   "Error = predicted − actual; positive means overestimated remaining life.")
        for f in ("pred_vs_actual", "test_trajectories", "residuals", "loss_curves", "baseline_comparison"):
            p = REPORTS / "figures" / f"{f}.png"
            if p.exists():
                st.image(str(p), caption=f.replace("_", " "))
        for extra in ("test_per_unit_selected_rnn.csv", "ablation_sequence_length.csv", "training_runs.csv"):
            p = REPORTS / "tables" / extra
            if p.exists():
                st.subheader(extra.removesuffix(".csv").replace("_", " "))
                st.dataframe(pd.read_csv(p), hide_index=True)
    st.subheader("Limitations")
    st.markdown((ROOT / "docs" / "limitations.md").read_text(encoding="utf-8"))


