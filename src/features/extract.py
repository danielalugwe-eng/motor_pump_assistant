from __future__ import annotations

from typing import Any

import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.stats import kurtosis, skew

FS = 12000


def extract_features(window: np.ndarray) -> dict[str, float]:
    """Compute a compact set of vibration features from a signal window."""
    window = np.asarray(window, dtype=float).reshape(-1)
    if window.size == 0:
        raise ValueError("window must not be empty")

    feats: dict[str, float] = {}
    feats["mean"] = float(np.mean(window))
    feats["std"] = float(np.std(window))
    feats["rms"] = float(np.sqrt(np.mean(window**2)))
    feats["skew"] = float(skew(window))
    feats["kurtosis"] = float(kurtosis(window))
    feats["ptp"] = float(np.ptp(window))
    feats["crest_factor"] = float(np.max(np.abs(window)) / (feats["rms"] + 1e-9))

    freqs = rfftfreq(len(window), 1 / FS)
    fft_vals = np.abs(rfft(window))
    feats["dominant_freq"] = float(freqs[np.argmax(fft_vals)])
    feats["spectral_energy"] = float(np.sum(fft_vals**2))
    feats["spectral_entropy"] = float(
        -np.sum((fft_vals / np.sum(fft_vals)) * np.log(fft_vals / np.sum(fft_vals) + 1e-12))
    )
    return feats
