from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .extract import extract_features

FS = 12000
WINDOW_SIZE = 2048
OVERLAP = 0.5
STEP = int(WINDOW_SIZE * (1 - OVERLAP))


def get_label(filename: Path) -> str | None:
    name = filename.stem
    if "Normal" in name:
        return "Normal"
    if "_IR_" in name:
        return "InnerRace"
    if "_B_" in name:
        return "Ball"
    if "OR@6" in name:
        return "OuterRace"
    return None


def window_signal(signal: np.ndarray, window_size: int = WINDOW_SIZE, step: int = STEP) -> list[np.ndarray]:
    windows: list[np.ndarray] = []
    for start in range(0, len(signal) - window_size, step):
        windows.append(signal[start : start + window_size])
    return windows


def build_windows(data_dir: str | Path | None = None) -> pd.DataFrame:
    data_dir = Path(data_dir or Path("data/raw/CWRU_Bearing_NumPy/Data"))
    files = list(data_dir.glob("**/*_7_DE12.npz")) + list(data_dir.glob("**/*_Normal.npz"))

    rows: list[dict[str, Any]] = []
    for file_path in files:
        label = get_label(file_path)
        if label is None:
            continue

        with np.load(file_path) as data:
            signal = data["DE"]

        if np.std(signal) < 1e-6:
            continue

        for chunk in window_signal(signal):
            if np.std(chunk) < 1e-6:
                continue
            rows.append({"window": chunk, "label": label, "source_file": file_path.name})

    df = pd.DataFrame(rows)
    if not df.empty:
        output_path = Path("data/processed/windows.pkl")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_pickle(output_path)
    return df


def build_feature_table(input_path: str | Path | None = None) -> pd.DataFrame:
    input_path = Path(input_path or Path("data/processed/windows.pkl"))
    df = pd.read_pickle(input_path)
    feature_rows = [extract_features(window) for window in df["window"]]
    features_df = pd.DataFrame(feature_rows)
    features_df["label"] = df["label"].values
    output_path = Path("data/processed/features.parquet")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_parquet(output_path, index=False)
    return features_df
