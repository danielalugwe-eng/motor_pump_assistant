from __future__ import annotations

from pathlib import Path

import pandas as pd
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report


def run_drift_check(reference: str | Path | None = None, current: str | Path | None = None, output_path: str | Path | None = None) -> None:
    reference_path = Path(reference or Path("data/processed/features.parquet"))
    current_path = Path(current or Path("data/processed/features.parquet"))
    output_path = Path(output_path or Path("reports/drift_report.html"))

    ref_df = pd.read_parquet(reference_path)
    current_df = pd.read_parquet(current_path)

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref_df.drop(columns=["label"]), current_data=current_df.drop(columns=["label"]))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.save_html(output_path)
    print(f"Drift report saved to {output_path}")
