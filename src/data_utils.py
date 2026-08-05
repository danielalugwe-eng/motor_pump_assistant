from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.features.extract import extract_features
from src.features.pipeline import WINDOW_SIZE, window_signal


def load_latest_sensor_data(path: str | Path | None = None) -> pd.DataFrame:
    path = Path(path or Path("data/processed/latest_week.csv"))
    if not path.exists():
        raise FileNotFoundError(f"Incoming sensor data file not found: {path}")
    return pd.read_csv(path)


def build_breakdown_signal(features: pd.DataFrame) -> dict[str, float]:
    risk_score = float(features["rms"].mean() * 0.4 + features["kurtosis"].mean() * 0.6)
    return {"risk_score": risk_score, "status": "high_risk" if risk_score > 1.0 else "normal"}


def load_cwru_npz_signal(npz_bytes: bytes, signal_key: str | None = None) -> tuple[np.ndarray, str]:
    """Load a vibration signal from a CWRU-style .npz payload."""
    with np.load(BytesIO(npz_bytes), allow_pickle=False) as data:
        if not data.files:
            raise ValueError("The uploaded .npz file does not contain any arrays.")

        preferred_keys = [signal_key] if signal_key else []
        preferred_keys += ["DE", "FE", "BA"]

        chosen_key = next((key for key in preferred_keys if key and key in data.files), data.files[0])
        signal = np.asarray(data[chosen_key], dtype=float).reshape(-1)

    if signal.size == 0:
        raise ValueError("The uploaded vibration array is empty.")

    return signal, chosen_key


def extract_vibration_features_from_npz(npz_bytes: bytes, signal_key: str | None = None) -> dict[str, Any]:
    """Convert a raw CWRU .npz vibration file into averaged model features."""
    signal, chosen_key = load_cwru_npz_signal(npz_bytes, signal_key=signal_key)

    windows = window_signal(signal) if signal.size >= WINDOW_SIZE else [signal]
    feature_rows: list[dict[str, float]] = []
    for window in windows:
        if np.std(window) < 1e-6:
            continue
        feature_rows.append(extract_features(window))

    if not feature_rows:
        feature_rows.append(extract_features(signal))

    features_df = pd.DataFrame(feature_rows)
    feature_row = features_df.mean(numeric_only=True).to_dict()
    feature_row["signal_key"] = chosen_key
    feature_row["window_count"] = len(feature_rows)
    feature_row["raw_signal_length"] = int(signal.size)
    return feature_row
