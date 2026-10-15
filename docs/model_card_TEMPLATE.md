# Model Card (template; fill only from generated artifacts and tables)

- **Model**: compact LSTM regressor, version `<model_version>`, selected seed `<seed>` (chosen on validation loss)
- **Architecture**: Input(L, F) → LSTM(units) → Dropout → Dense(16, ReLU) → Dense(1, softplus); parameters `<n>`
- **Inputs**: `<feature_names>` (measured channels + elapsed time); sequence length `<L>` `<time unit>` steps
- **Target**: RUL in `<time unit>` = documented failure time − current time; training cap `<cap or none>`
- **Training data**: units `<train ids>`; validation units `<val ids>`; preprocessing fitted on training units only
- **Evaluation**: held-out units `<test ids>`; see `reports/tables/test_comparison.csv`
- **Intended use**: coursework demonstration of sequence regression on recorded data
- **Not intended for**: maintenance, dispatch or airworthiness decisions; equipment other than the dataset's
- **Limitations**: see `docs/limitations.md`
