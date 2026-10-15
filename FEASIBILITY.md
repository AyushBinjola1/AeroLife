# Dataset Feasibility Gate

**Date of investigation:** 2026-10-15
**Requirement under test:** a publicly available, **genuinely measured** turbofan
multi-sensor time-series dataset with defensible end-of-life labels for supervised
RUL regression. Measured inputs are required, and so are measured degradation
trajectories. Simulated sensors, simulated degradation and invented labels are not
accepted.

## Verdict

> **An exact dataset satisfying both the turbofan assignment and the real-measurement-only requirement has not been verified.**

Every public turbofan run-to-failure dataset found comes from a NASA C-MAPSS
simulation. The real aviation datasets found either have no end-of-life events
(flight-recorder data) or contain only seeded-fault classes (aero-engine
bearing diagnosis data). Neither kind supports supervised RUL regression.

The project has **not** been changed. Training and evaluation stay **blocked**
until faculty record one of the decisions below in `configs/config.yaml`
(`dataset.decision`, `approved_by`, `approved_on`).

## Faculty decision required

| | Decision A | Decision B |
|---|---|---|
| Keeps | Aircraft-engine (turbofan) topic | Real-measurement-only requirement |
| Changes | Allows a **clearly disclosed simulation benchmark** (C-MAPSS or N-CMAPSS) | Equipment becomes **rotating machinery (rolling-element bearings)** |
| Candidate | N-CMAPSS (real recorded flight conditions, simulated engine response and degradation) or C-MAPSS FD001–FD004 | XJTU-SY bearing accelerated life tests |
| Required wording | "simulated turbofan data (NASA C-MAPSS model)" on every screen and in the report; never "real engine data" | Project renamed to bearing RUL everywhere; no claim of transfer to aircraft engines |
| Time unit | Flight cycles (C-MAPSS) / flights + 1 Hz samples within flights (N-CMAPSS) | Minutes between vibration recordings (accelerated test, not service hours) |
| Open issues | None on availability; the realism limitation must be disclosed | The end-of-life definition must be confirmed from the primary paper (see below). Download hosts are blocked in the current cloud container |

Neither substitution is implemented in this repository. Both need explicit
authorisation, and the current instruction does not authorise simulated data.

## Source-backed feasibility table

How much was verified: this container's network policy allowed only GitHub and
PyPI. The publisher sites (nasa.gov, mdpi.com, ieee.org, phmsociety.org,
Google Drive, Dropbox) returned HTTP 403 at the proxy. So the evidence below comes from:
**[R]** primary text read in full; **[S]** quoted excerpts of primary or
peer-reviewed secondary sources returned by web search; **[U]** not verified.
Before any decision is acted on, re-check the [S] and [U] items against the
original documents.

| Dataset | Physical equipment & collection | Sensors measured or simulated? | Units | Sensors / units / sampling | Event definition & labels | Censoring | Access / terms / citation | Verdict |
|---|---|---|---|---|---|---|---|---|
| **NASA C-MAPSS** (Turbofan Engine Degradation Simulation) | None. Engines simulated with the C-MAPSS model in MATLAB/Simulink [S: Saxena et al. 2008] | **Simulated.** Readme: "The data is contaminated with sensor noise"; damage propagation is modelled [R: readme text; S: paper] | FD001 100/100, FD002 260/259, FD003 100/100, FD004 248/249 train/test [R] | 3 operational settings + 21 sensor channels; one row per operational cycle [R]. Channel units not in readme [U] | Training trajectories run to simulated failure; test truncated, true RUL in `RUL_FD00x.txt` [R] | Test units truncated by design, with labels supplied [R] | Public, NASA PCoE repository; cite Saxena & Goebel 2008 data set and Saxena et al. PHM08 paper | **Reject** for real-only. Candidate for Decision A |
| **N-CMAPSS** (Arias Chao et al., *Data* 6(1):5, 2021) | None physical. CMAPSS model driven by **real flight conditions recorded on a commercial jet** [S] | **Mixed.** Scenario descriptors *w* are real recordings. Sensor outputs *x_s*, virtual sensors *x_v* and health parameters θ come from the model; degradation is simulated [S] | 128 units across 8 subsets [S, secondary] | *w*, *x_s*, *x_v*, θ, auxiliary data; per-second samples within flights [U] | Complete run-to-failure; RUL labels derived from the simulated end of life [S] | None (full histories) [S] | Public, NASA PCoE; paper CC BY 4.0 [S] | **Reject** for real-only (sensor values and degradation are simulated). Strongest candidate for Decision A |
| **PHM 2008 / PHM 2021 challenge data** | Same as C-MAPSS / subset of N-CMAPSS [S] | Simulated [S] | 100 aircraft (2021 challenge) [S] | as above | as above | as above | PHM Society | **Reject** (same provenance as above) |
| **HIT inter-shaft bearing dataset** (Hou et al., Harbin Inst. of Technology) | Real aero-engine test rig, dual rotor; 2 displacement sensors + 4 casing accelerometers [S] | Measured [S] | Not individual run-to-failure units | 28 speed combinations; 25 kHz (secondary) [S] | **Seeded faults, classes only** (healthy / inner / outer race) [S] | n/a | GitHub `HouLeiHIT/HIT-dataset` → Google Drive [S] | **Reject**: fault classes only, no run-to-failure, no RUL |
| **NASA DASHlink flight-recorder data** (Sample Flight Data, FOQA) | Real airline flight-recorder parameters, released for aviation-safety data mining [S] | Measured [S] | Flights, not engines run to failure | Many parameters [U] | **No engine end-of-life events found** in any description [S] | n/a | Public via DASHlink [S] | **Reject**: unrelated flight observations, no defensible RUL labels |
| **XJTU-SY bearings** (Wang, Lei, Li, Li, *IEEE Trans. Reliability* 69(1):401-412, 2020) | Real accelerated degradation tests of 15 LDK UER204 rolling bearings on the XJTU / Sumyoung rig [R: README; S: setup] | **Measured** vibration (PCB 352C33 accelerometers, horizontal and vertical) [S] | 15 bearings: 5 per condition [R/S] | C1 2100 rpm / 12 kN, C2 2250 rpm / 11 kN, C3 2400 rpm / 10 kN [S]. Waveform 25.6 kHz, 32,768 samples (1.28 s) **per recording, one recording per minute** [S]. One reports 25 kHz [S] | "Complete run-to-failure data" [R]. Tests stop at a vibration-amplitude threshold, but **sources conflict**: 10 × A_h (A_h = healthy-stage maximum amplitude) vs 20 g [S] | No censoring stated [R] | Six download mirrors (Google Drive, Dropbox, MediaFire, MEGA, Baidu, author site) [R]. **No formal licence**; users "requested to cite" the 2020 paper [R] | **Candidate for Decision B**, pending primary confirmation of the end-of-life criterion and file format |

### Sources

- NASA C-MAPSS readme (text read in full from a GitHub mirror, `edwardzjl/CMAPSSData/readme.txt`; check against the NASA PCoE original).
- A. Saxena, K. Goebel, D. Simon, N. Eklund, "Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation," PHM08, 2008.
- M. Arias Chao, C. Kulkarni, K. Goebel, O. Fink, "Aircraft Engine Run-to-Failure Dataset under Real Flight Conditions for Prognostics and Diagnostics," *Data* 6(1):5, 2021. doi:10.3390/data6010005.
- PHM Society 2021 Data Challenge description (data.phmsociety.org).
- XJTU-SY README, github.com/WangBiaoXJTU/xjtu-sy-bearing-datasets (read in full, 2026-10-15).
- B. Wang, Y. Lei, N. Li, N. Li, "A Hybrid Prognostics Approach for Estimating Remaining Useful Life of Rolling Element Bearings," *IEEE Trans. Reliability* 69(1):401-412, 2020. doi:10.1109/TR.2018.2882682 (full text not retrieved).
- Secondary descriptions of the XJTU-SY setup: arXiv 2405.01614 (RULSurv), arXiv 2109.12513. The 10 × A_h criterion appears in a USPTO patent text that cites the dataset.
- HIT dataset: L. Hou et al., inter-shaft bearing fault diagnosis dataset from an aero-engine (scholar.hit.edu.cn); arXiv 2506.16225.
- NASA DASHlink Sample Flight Data project (c3.ndc.nasa.gov/dashlink/projects/85).

## What must happen before XJTU-SY (Decision B) can be used

1. Faculty records Decision B in `configs/config.yaml`.
2. Rename the project and every screen and report to bearing RUL. Example title: *"Remaining Useful Life Estimation of Rolling-Element Bearings from Measured Vibration Time Series"*.
3. Download the archive from a publisher mirror on a network that allows it. Place it unmodified in `data/raw/` and record the retrieval date, URL and SHA-256 (`scripts/ingest.py` writes the manifest).
4. Confirm from Wang et al. (2020) or the accompanying tutorial which stop and failure criterion was used (10 × A_h or 20 g), the file layout (CSV per minute, column order) and the sampling rate. Write all of it in `docs/dataset_card.md`.
5. Write the adapter (`src/aerolife/adapters/`):
   - one observation per recording;
   - `time_value` = minutes since test start (= recording index × 1 min, if confirmed);
   - `failure_time` = time of the final recording only if the source says the test stopped at the failure threshold;
   - per-recording features from `aerolife.features.burst_features`.
6. With 15 units, use grouped holdout (e.g. 1 test bearing per condition) plus leave-one-unit-out validation. Report per-unit errors.

## What Decision A would require

- Every page and the report must show: *"Simulated turbofan data generated by NASA's C-MAPSS model; not measured engine data."*
- Use the official train/test files and the official `RUL_*.txt` test labels for C-MAPSS, or the dev/test split for N-CMAPSS.
- Never use the N-CMAPSS health parameters θ, virtual sensors or auxiliary health state as inputs.
