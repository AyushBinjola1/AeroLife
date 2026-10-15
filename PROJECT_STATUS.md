# Project Status (2026-10-15)

## Blocker

**An exact dataset satisfying both the turbofan assignment and the real-measurement-only
requirement has not been verified.** See `FEASIBILITY.md`.

The blocking decision belongs to faculty: **A** (turbofan topic + disclosed simulation benchmark)
or **B** (real measured data + change to bearings, e.g. XJTU-SY). Neither is authorised.

Secondary blocker for Decision B in this cloud container: the network policy denies the XJTU-SY
download hosts (Google Drive, Dropbox, MediaFire, MEGA, Baidu, biaowang.tech) and the publisher
paper sites. The files must be downloaded on a machine that can reach them, or the hosts must be
allowed in the environment's network settings.

## Completed and verified

- Feasibility investigation with sources and verification levels.
- Dataset-agnostic library: schema validation, RUL targets with censoring, unit-level splits with
  manifest, causal preprocessing, vibration features, per-unit windowing, four baselines,
  LSTM/GRU/SimpleRNN builder and trainer, metrics (unit-averaged and pooled, early vs near-failure,
  over-estimation, unit bootstrap, PHM08 score), saved-artifact inference, reporting.
- 38 passing tests (`python -m pytest -q`). They cover unit separation, window boundaries, causal
  preprocessing, target alignment, schema validation, saved-model inference and the dashboard empty
  states. The end-to-end test runs the full pipeline on a hand-written logic fixture, only to check
  mechanics.
- Streamlit dashboard with six pages and honest empty states.
- Documentation: data policy, dataset/model card templates, report (results sections marked
  blocked), slides outline, 36 viva questions, demo script, contribution statement, UCS321 mapping.

## Not done (blocked; do not claim)

- Dataset adapter, ingestion, provenance manifest of real files.
- Split manifest for real units.
- Any trained model, any metric, any figure, the ablation, and the 5-minute demo on recorded data.
- Dataset card and model card contents.
