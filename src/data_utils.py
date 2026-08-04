from __future__ import annotations

import os
from pathlib import Path

import pandas as pd


def load_latest_sensor_data(path: str | Path | None = None) -> pd.DataFrame:
    path = Path(path or Path("data/processed/latest_week.csv"))
    if not path.exists():
        raise FileNotFoundError(f"Incoming sensor data file not found: {path}")
    return pd.read_csv(path)


def build_breakdown_signal(features: pd.DataFrame) -> dict[str, float]:
    risk_score = float(features["rms"].mean() * 0.4 + features["kurtosis"].mean() * 0.6)
    return {"risk_score": risk_score, "status": "high_risk" if risk_score > 1.0 else "normal"}
