from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

SELECTED_FEATURES = ["rms", "kurtosis", "crest_factor", "dominant_freq", "spectral_energy", "spectral_entropy"]


def train_classifier(features_path: str | Path | None = None, model_path: str | Path | None = None) -> tuple[RandomForestClassifier, dict[str, object]]:
    features_path = Path(features_path or Path("data/processed/features.parquet"))
    model_path = Path(model_path or Path("models/mixer_fault_classifier_v1.pkl"))

    df = pd.read_parquet(features_path)
    X = df[SELECTED_FEATURES]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    clf = RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")
    cv_scores = cross_val_score(clf, X_train, y_train, cv=StratifiedKFold(5))

    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, model_path)

    results = {
        "cv_accuracy": float(cv_scores.mean()),
        "classification_report": classification_report(y_test, preds, output_dict=True),
    }
    return clf, results


def main() -> None:
    _, results = train_classifier()
    print(results["cv_accuracy"])
    print(results["classification_report"])


if __name__ == "__main__":
    main()
