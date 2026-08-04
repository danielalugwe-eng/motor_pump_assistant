from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

SELECTED_FEATURES = ["rms", "kurtosis", "crest_factor", "dominant_freq", "spectral_energy", "spectral_entropy"]


def build_models() -> dict[str, Any]:
    return {
        "random_forest": RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced"),
        "gradient_boosting": GradientBoostingClassifier(random_state=42),
        "svm": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", SVC(kernel="rbf", probability=True, random_state=42)),
            ]
        ),
    }


def evaluate_models(features_path: str | Path | None = None) -> dict[str, Any]:
    features_path = Path(features_path or Path("data/processed/features.parquet"))
    df = pd.read_parquet(features_path)
    X = df[SELECTED_FEATURES]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    results: dict[str, Any] = {}

    for name, model in build_models().items():
        cv_scores = cross_val_score(model, X_train, y_train, cv=StratifiedKFold(5))
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        results[name] = {
            "cv_accuracy": float(cv_scores.mean()),
            "test_report": classification_report(y_test, preds, output_dict=True),
        }

    return results


def train_best_model(features_path: str | Path | None = None, model_path: str | Path | None = None) -> tuple[str, Any, dict[str, Any]]:
    results = evaluate_models(features_path)
    best_name = max(results, key=lambda name: results[name]["cv_accuracy"])
    model = build_models()[best_name]

    features_path = Path(features_path or Path("data/processed/features.parquet"))
    df = pd.read_parquet(features_path)
    X = df[SELECTED_FEATURES]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    model.fit(X_train, y_train)
    model_path = Path(model_path or Path("models/best_fault_classifier.pkl"))
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)

    return best_name, model, results


def main() -> None:
    best_name, _, results = train_best_model()
    print(f"Best model: {best_name}")
    print(results[best_name])


if __name__ == "__main__":
    main()
