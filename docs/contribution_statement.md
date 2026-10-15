# Contribution Statement

## Implemented in this project
- Dataset feasibility investigation and the decision framework (`FEASIBILITY.md`).
- All code in `src/`, `scripts/`, `app/` and `tests/`: schema validation, target construction with censoring, unit-level
  splitting, causal preprocessing, vibration features, windowing, baselines, model builder and trainer, metrics,
  unit bootstrap, saved-artifact inference, reporting, and the Streamlit dashboard.
- Documentation in `docs/`.

## Taken from public sources
- Dataset descriptions, cited in `FEASIBILITY.md`. No dataset files are included.
- Standard methods: LSTM (Hochreiter & Schmidhuber, 1997), GRU (Cho et al., 2014), the PHM08 scoring function
  (Saxena et al., 2008), standard vibration statistics (RMS, kurtosis, crest factor).
- Open-source libraries listed in `requirements.txt` (TensorFlow/Keras, scikit-learn, pandas, NumPy, SciPy,
  Streamlit, Plotly, Matplotlib).

## Assistance
Code and documents were drafted with an AI coding assistant under the team's direction. The team is
responsible for reviewing, running and being able to explain every part. *(Edit this section so it describes
each team member's individual contribution accurately.)*
