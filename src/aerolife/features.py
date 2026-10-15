"""Per-recording features for high-frequency measured vibration.

Each measurement burst (one recording) becomes one observation row, so the
time step of the sequence model is the interval between recordings, NOT the
waveform sampling period. Formulas (x = one recording, N samples):

    RMS            sqrt(mean(x^2))
    std            sqrt(mean((x - mean(x))^2))
    peak_to_peak   max(x) - min(x)
    kurtosis       mean((x-mu)^4) / std^4        (Pearson; 3 for a Gaussian)
    skewness       mean((x-mu)^3) / std^3
    crest_factor   max|x| / RMS
    spectral_centroid  sum(f * P(f)) / sum(P(f)), P = |rFFT(x)|^2
    band_energy_k  sum_{f in band k} P(f) / sum(P(f))

Zero-division safeguard: any ratio whose denominator is below EPS is set to 0
(a flat recording has no shape, so its shape statistics are reported as 0).
"""

from __future__ import annotations

import numpy as np

EPS = 1e-12


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if abs(b) > EPS else 0.0


def burst_features(x: np.ndarray, fs: float, bands_hz: list[tuple[float, float]] | None = None) -> dict:
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or x.size < 2:
        raise ValueError("expected a 1-D recording with at least 2 samples")
    mu = x.mean()
    d = x - mu
    var = np.mean(d**2)
    sd = np.sqrt(var)
    rms = np.sqrt(np.mean(x**2))
    feats = {
        "rms": float(rms),
        "std": float(sd),
        "peak_to_peak": float(x.max() - x.min()),
        "kurtosis": _safe_div(np.mean(d**4), var**2),
        "skewness": _safe_div(np.mean(d**3), sd**3),
        "crest_factor": _safe_div(np.max(np.abs(x)), rms),
    }
    power = np.abs(np.fft.rfft(d)) ** 2
    freqs = np.fft.rfftfreq(x.size, d=1.0 / fs)
    total = power.sum()
    feats["spectral_centroid"] = _safe_div((freqs * power).sum(), total)
    for lo, hi in bands_hz or []:
        m = (freqs >= lo) & (freqs < hi)
        feats[f"band_{int(lo)}_{int(hi)}"] = _safe_div(power[m].sum(), total)
    return feats
