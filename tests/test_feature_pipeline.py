import numpy as np

from src.features.extract import extract_features


def test_extract_features_returns_expected_keys():
    window = np.sin(np.linspace(0, 4 * np.pi, 2048))
    features = extract_features(window)

    assert set(features.keys()) == {
        "mean",
        "std",
        "rms",
        "skew",
        "kurtosis",
        "ptp",
        "crest_factor",
        "dominant_freq",
        "spectral_energy",
        "spectral_entropy",
    }
    assert np.isfinite(features["dominant_freq"])
    assert np.isfinite(features["spectral_energy"])
