# AeroLife: Remaining Useful Life Estimation from Multi-Sensor Time Series

UCS321 project. **Status: dataset gate BLOCKED. No model has been trained and there are no results.**

The assignment context is a Rolls-Royce Aerospace brief asking for an RNN/LSTM that predicts
turbofan Remaining Useful Life (RUL). This project has no affiliation with Rolls-Royce, no
access to its data, and no operational validation.

## Current state

| Component | State |
|---|---|
| Dataset feasibility gate (`FEASIBILITY.md`) | Done. Verdict: no public turbofan dataset with real measured degradation was verified |
| Faculty decision (A: disclosed simulation, B: real bearing data) | **Pending** |
| Canonical schema + validation (`src/aerolife/schema.py`) | Done, tested |
| RUL targets, censoring, optional cap (`targets.py`) | Done, tested |
| Unit-level splits + manifest, leave-one-unit-out (`splits.py`) | Done, tested |
| Causal preprocessing fitted on training units (`preprocessing.py`) | Done, tested |
| Vibration burst features (`features.py`) | Done, tested |
| Per-unit sliding windows, insufficient-history rule (`windowing.py`) | Done, tested |
| Baselines (constant, age-based, Ridge, Random Forest) + LSTM/GRU/SimpleRNN | Done; mechanics tested on a logic fixture only |
| Metrics, unit bootstrap, PHM08 score, report figures (`metrics.py`, `reporting.py`) | Done, tested |
| Saved-artifact inference (`inference.py`) | Done, round-trip tested |
| Streamlit dashboard, 6 pages, honest empty states | Done, tested with `streamlit.testing` |
| Dataset adapter (raw files → canonical tables) | **Blocked** (dataset-specific) |
| Training, evaluation, figures, ablation, demo | **Blocked** |

Run `python scripts/status.py` at any time to see this from the files on disk.

## Setup (verified on Python 3.13, Linux, CPU)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q          # 38 tests
streamlit run app/streamlit_app.py
```

## Pipeline (each step refuses to run while the gate is blocked; exit code 2)

```bash
python scripts/status.py
python scripts/ingest.py         # raw → data/canonical/{observations,units}.csv + provenance manifest
python scripts/make_splits.py    # units → train/val/test BEFORE windowing; data/splits/split_manifest.json
python scripts/train.py          # baselines + LSTM (3 seeds); artifacts/v<version>/
python scripts/evaluate.py       # reports/tables/*.csv, reports/figures/*.png from saved predictions
python scripts/ablation.py --lengths 10 20 30 50   # selection on validation units only
python scripts/predict.py my_history.csv --out preds.csv   # saved model + preprocessing only
```

## Unblocking the project

1. Faculty choose Decision A or B (`FEASIBILITY.md`). Record it in `configs/config.yaml`
   (`gate_status: approved`, `decision`, `approved_by`, `approved_on`, `name`, `spec_path`).
2. Copy `docs/dataset_spec_TEMPLATE.yaml` → `configs/dataset_spec.yaml` and fill it from the
   publisher documentation.
3. Download the raw files from the publisher into `data/raw/` without modifying them (`docs/data_policy.md`).
4. Implement and register one adapter in `src/aerolife/adapters/`.
5. If Decision B: rename the project and every page and report to bearing RUL.
6. Run the pipeline above. Fill `docs/dataset_card.md`, `docs/model_card.md` and the results
   sections of `docs/report.md` **from the generated tables only**.

## Repository layout

```
FEASIBILITY.md            dataset gate: source-backed table + faculty decision
PROJECT_STATUS.md         exact blockers and completed work
configs/config.yaml       all settings; dataset-dependent values are null
src/aerolife/             library (schema, targets, splits, preprocessing, features,
                          windowing, baselines, models, metrics, inference, pipeline, reporting)
src/aerolife/adapters/    dataset adapters (none registered: blocked)
scripts/                  CLI steps listed above
app/                      Streamlit dashboard (streamlit_app.py + views.py)
tests/                    38 tests: unit separation, window boundaries, causality, target
                          alignment, schema, saved-model inference, dashboard
notebooks/01_exploration.ipynb   EDA (shows the blocked state until data exists)
docs/                     data policy, templates, report, slides, viva, demo, contribution, UCS321 map
data/raw/                 immutable publisher files (empty)
artifacts/, reports/      generated outputs (empty)
```

## Honesty rules built into the code

- Scripts check the dataset gate and exit instead of running on substitute data.
- Label-like columns (`rul`, `failure_time`, `health_*`, `theta*`, …) are rejected as inputs and stripped from uploads.
- Preprocessing statistics come from training units only. Imputation is forward-only and rolling statistics are trailing-only.
- Windows never cross units. Units shorter than the sequence length get "insufficient history" and are never padded.
- The dashboard truncates a unit's history at the replay time before calling the model. Ground truth appears only in a separate retrospective panel.
- With no artifacts, every data page shows an empty state. Nothing is generated to fill it.

The test fixtures in `tests/conftest.py` are tiny hand-written integer tables used only to
check index logic. They live in temporary directories and never produce reported results.
