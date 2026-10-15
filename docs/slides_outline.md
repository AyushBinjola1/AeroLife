# Presentation Outline (11 slides)

1. **Title.** AeroLife RUL estimation. Team, UCS321. Disclaimer: assignment context only, no Rolls-Royce data or affiliation.
2. **Problem.** What RUL is and why it matters for proactive maintenance. RUL(t) = T_fail − t.
3. **Data feasibility gate.** The table from `FEASIBILITY.md`. Simulated vs measured. Why C-MAPSS and N-CMAPSS fail the real-only rule.
4. **Faculty decision.** Option A vs B, the one chosen, and the wording it imposes.
5. **Dataset and provenance.** Equipment, sensors, time-step meaning, units, end-of-life definition, censoring, checksums. *(after approval)*
6. **Leakage-safe pipeline.** Split by unit → fit on train → causal transforms → per-unit windows (diagram).
7. **Models.** Baselines; LSTM architecture and gates (one equation slide); training controls on validation units.
8. **Results.** Unit-averaged MAE/RMSE with bootstrap intervals, all models, same windows. *(measured only)*
9. **Error analysis.** Trajectories for held-out units, residuals, weakest unit, early vs near failure, overestimation.
10. **Dashboard demo.** Recorded-data replay, insufficient-history state, demonstration thresholds.
11. **Limitations and conclusions.** Few units, simulation or accelerated-test caveats, no transfer claim, future work.
