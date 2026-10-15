# 5-Minute Demonstration Script

**Status: BLOCKED.** The recorded held-out measurements needed for the demo do not exist until a
dataset is approved and the model is trained. Until then, the only honest demo is the empty-state
walkthrough (part A).

## A. Now (≈2 min)
1. `python scripts/status.py`: gate blocked, no artifacts.
2. `python scripts/train.py` exits with code 2 and prints the BLOCKED reason.
3. `streamlit run app/streamlit_app.py`. On the Overview, show the feasibility table and the decision banner.
   Click through the pages: each shows "unavailable" and nothing is fabricated.
4. RUL Prediction → upload a CSV. Schema validation runs and states that no model exists.
5. `python -m pytest -q`: 38 passing tests. Name the leakage checks.

## B. After approval and training (5 min)
| Time | Action | Say |
|---|---|---|
| 0:00 | Overview page | Dataset provenance, decision taken, time unit |
| 0:45 | Fleet Explorer | Units, conditions, which units are held out |
| 1:30 | Sensor History, held-out unit, slider at an early step | "Recorded-data replay; only data up to this point is shown" |
| 2:15 | RUL Prediction at step < L | "Insufficient history: we refuse rather than pad" |
| 2:45 | Advance the slider through life | Predicted RUL vs retrospective actual in the separate panel |
| 3:30 | Maintenance demo | Configurable thresholds, alert time vs actual remaining life; not dispatch criteria |
| 4:15 | Evaluation page | Baseline table with intervals, weakest unit, limitations |
