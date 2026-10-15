# UCS321 Mapping

| Syllabus area | Where in the project |
|---|---|
| Engineering AI problem framing | `FEASIBILITY.md`, `docs/report.md` §1–3: requirement analysis, data provenance gate, decision record |
| Sensor / time-series preprocessing | `preprocessing.py` (causal fill, trailing rolling mean, train-only scaling, constant-channel audit), `features.py` (vibration burst features) |
| Supervised regression | `targets.py` (RUL labels, censoring, cap), `baselines.py` (constant, age, Ridge, Random Forest) |
| Neural networks | `models.py`: dense head, activations, Adam, loss choice, regularisation (dropout, early stopping) |
| RNN / LSTM | `models.py` (LSTM, GRU, SimpleRNN), `windowing.py` (sequence construction); `viva_questions.md` §25–33 |
| Evaluation | `metrics.py`, `reporting.py`: unit-averaged vs pooled, bootstrap by unit, overestimation, ablation |
| Version control & reproducibility | git history, pinned `requirements.txt`, `configs/config.yaml`, split/provenance manifests, seeded runs, 38 tests |
