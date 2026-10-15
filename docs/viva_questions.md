# Viva Preparation: 36 Questions with Answers

## RUL and the problem

1. **What is Remaining Useful Life?**
   The time a unit is expected to keep operating, from now until it reaches a defined end-of-life
   criterion. It is measured in the dataset's own unit (cycles, minutes) and is only as meaningful
   as that criterion.

2. **Why is RUL estimation a regression problem and not classification?**
   The target is a continuous non-negative quantity. Classification ("healthy/faulty") discards
   how much time remains, and that number is what maintenance planning needs.

3. **How are training labels built?**
   For a unit whose failure time T is documented, RUL(t) = T − t at every observation time t.

4. **If labels use the future failure time, why is that not leakage?**
   Labels are retrospective supervision. They are computed after the run is complete, only to score
   the model. Leakage would be giving T (or anything derived from it, such as total run length or
   normalising by the whole lifetime) to the model as an input. At prediction time T is unknown, so
   the model sees only past and present measurements.

5. **Can you treat the last recorded observation as the failure time?**
   Only if the publisher says recording stopped at failure (for example, a test stopped when a
   vibration threshold was crossed). Otherwise the record may just end, and using its end as
   failure would invent a life.

6. **What is a censored trajectory and how is it handled?**
   A unit whose failure was not observed: it was still running when recording stopped. Its true RUL
   is unknown, only bounded below. The pipeline gives such units no labels. The schema rejects any
   failure time attached to them, and they are listed in the split manifest.

7. **What is an RUL cap and when is it justified?**
   Clipping the training target at a maximum, for example when early-life signals carry no
   degradation information. It must be chosen on validation units. Uncapped physical RUL must still
   be reported, because the capped target is not the real remaining life.

## Data, provenance and generalisation

8. **Why was C-MAPSS rejected?**
   Its sensor values and degradation are produced by a simulation model with added noise. The brief
   requires real measured data, and C-MAPSS cannot be described as real engine data.

9. **N-CMAPSS uses real flight data. Why is it still rejected?**
   Only the flight conditions (altitude, Mach number, and so on) were recorded on real aircraft.
   The engine sensor responses and the degradation are simulated by the CMAPSS model.

10. **Why not a real turbofan dataset?**
    None with documented end-of-life events could be verified as public. The real aviation datasets
    found have no failure events (flight recorders) or only seeded-fault classes (aero-engine bearing
    diagnosis).

11. **If bearings are approved, can you claim the model works on aircraft engines?**
    No. Different equipment, physics, failure modes and time scales. Accelerated laboratory tests
    also differ from service conditions. Any transfer is untested.

12. **What does one time step mean in each candidate dataset?**
    C-MAPSS: one operational cycle (flight). XJTU-SY: one vibration recording, 1.28 s of 25.6 kHz
    waveform, taken once per minute. The sequence step is the recording interval, not the waveform
    sampling period.

13. **Why not convert cycles or minutes to hours or days?**
    The source gives no conversion. A cycle has no fixed duration, and accelerated test minutes are
    not service time.

## Splits and leakage

14. **Why split by unit before windowing?**
    Overlapping windows from the same unit share almost all of their data. If they land in both
    train and test, the model is tested on what it has effectively memorised, and the scores come
    out optimistic.

15. **How many independent samples does the test set have?**
    The number of test units, not the number of windows. Metrics are averaged per unit first, and
    uncertainty is estimated by resampling units.

16. **What do you do when there are very few units?**
    Leave-one-unit-out or grouped validation for model choices, and a small fixed held-out test
    group. Per-unit results are reported, wide uncertainty is stated, and broad claims are avoided.

17. **Why fit the scaler only on training units?**
    Statistics computed from validation or test units carry information about them into training,
    which is leakage. Normalising a held-out unit by its own complete lifetime would also use its
    future.

18. **What does "causal preprocessing" mean here?**
    Every transform at time t uses only data at or before t: forward-fill imputation, trailing
    rolling means, fixed training statistics. A test checks that changing future values leaves past
    outputs unchanged.

19. **Why is centred smoothing not allowed?**
    A centred window at time t averages values after t, which would not exist at prediction time.

20. **How do operating conditions affect the split?**
    If a condition appears in training and test, the test measures interpolation. If a test
    condition is absent from training, it measures transfer. The manifest records conditions per
    partition so the claim can be stated correctly.

## Sequence windows and features

21. **What is a sliding window and what is its label?**
    The last L observations of one unit, labelled with the RUL at the window's final step, which is
    the moment the prediction is made.

22. **What happens when a unit has fewer than L observations?**
    The system reports "insufficient history" and makes no prediction. No padding is used, so no
    made-up values look like observations.

23. **How are sequence length and stride chosen?**
    On validation units only, for example through the sequence-length ablation. The test set is
    never used to choose them.

24. **Which vibration features would you use and why?**
    RMS (energy), standard deviation, peak-to-peak, kurtosis (impulsiveness from early defects),
    crest factor (peak relative to RMS), spectral centroid and band energies. Each is computed per
    recording, and ratios return 0 when the denominator is ~0.

## RNN, LSTM and training

25. **What is an RNN?**
    A network that processes a sequence one step at a time, updating a hidden state
    h_t = f(W x_t + U h_{t−1} + b) that summarises the past.

26. **Why can a simple RNN struggle with long sequences?**
    Back-propagated gradients are multiplied by the recurrent Jacobian at every step, so they shrink
    or explode. The network then fails to learn long-range dependencies.

27. **What are the LSTM gates?**
    Forget gate f_t decides what to erase from the cell state. Input gate i_t decides what new
    candidate information to write. Output gate o_t decides what part of the cell state to expose
    as h_t. Update: c_t = f_t ⊙ c_{t−1} + i_t ⊙ c̃_t, then h_t = o_t ⊙ tanh(c_t).

28. **Why does the cell state help?**
    Its additive update gives gradients a path that is not repeatedly squashed. Information can
    persist across many steps when the forget gate stays near 1.

29. **How does a GRU differ?**
    It merges forget and input into an update gate, adds a reset gate, and has no separate cell
    state. It has fewer parameters and often performs similarly.

30. **Why a compact model (32 units)?**
    With few independent units, large models overfit. A small model is easier to explain, trains
    on a CPU, and should be enlarged only if validation evidence supports it.

31. **Why Huber loss?**
    Quadratic for small errors and linear for large ones. That makes it less sensitive than MSE to
    a few very large early-life errors, while staying smooth near zero, unlike MAE.

32. **What do early stopping, LR reduction and checkpointing monitor?**
    Validation-unit loss only. The test units are untouched until final evaluation.

33. **How is the output kept non-negative?**
    A softplus output activation. RUL cannot be negative.

## Evaluation

34. **Why report both unit-averaged and pooled metrics?**
    Pooled metrics weight every window equally, so long-lived units dominate. Unit-averaged metrics
    treat each independent unit equally. Both are reported, with the weighting stated.

35. **Why does overestimation matter more?**
    Predicting more life than remains can delay maintenance past failure. The overestimation
    fraction and magnitude are reported separately. The PHM08 score penalises late predictions more
    (exp(d/10) vs exp(−d/13)), but its constants were defined for C-MAPSS cycles.

36. **How would you decide whether the LSTM beats the baselines?**
    Compare models on identical units, windows, targets and evaluation times, across several seeds,
    with unit-bootstrap intervals. If the intervals overlap heavily, or a simple baseline is as good,
    say so. Superiority is claimed only if the measured results show it.
