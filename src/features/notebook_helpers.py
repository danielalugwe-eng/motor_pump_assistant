from __future__ import annotations

from pathlib import Path

import pandas as pd

from .pipeline import build_feature_table, build_windows


def prepare_data() -> pd.DataFrame:
    build_windows()
    return build_feature_table()


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]
