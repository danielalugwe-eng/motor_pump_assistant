from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.features.model_training import train_best_model
from src.monitoring.drift import run_drift_check


def retrain_if_needed(reference_path: str | Path | None = None, current_path: str | Path | None = None) -> dict[str, object]:
    reference_path = Path(reference_path or Path("data/processed/features.parquet"))
    current_path = Path(current_path or Path("data/processed/features.parquet"))

    ref_df = pd.read_parquet(reference_path)
    cur_df = pd.read_parquet(current_path)

    if ref_df.shape[0] == cur_df.shape[0] and ref_df.equals(cur_df):
        print("No new data detected; skipping retraining.")
        return {"status": "skipped", "reason": "no_new_data"}

    run_drift_check(reference_path, current_path)
    best_name, _, results = train_best_model(current_path)
    return {"status": "retrained", "best_model": best_name, "results": results}


def main() -> None:
    print(retrain_if_needed())


if __name__ == "__main__":
    main()
