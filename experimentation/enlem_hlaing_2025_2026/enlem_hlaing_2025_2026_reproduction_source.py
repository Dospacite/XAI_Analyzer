#!/usr/bin/env python3
"""Notebook source for reproducing the EnLeM phishing classifier.

No official training code was found for the paper. This notebook implements a
paper-aligned local reproduction on the public UCI ARFF dataset using:

- ARFF preparation
- feature cleaning and label mapping
- 10-fold CV
- mutual-information SelectKBest
- hard-voting ensemble of Decision Tree, Random Forest, and k-NN
"""

from __future__ import annotations

# %%!
import subprocess
import sys

INSTALL_DEPS = False

if INSTALL_DEPS:
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--upgrade",
            "numpy",
            "pandas",
            "scikit-learn",
        ]
    )

# %%!
import json
import pickle
import re
from collections import Counter
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# %%!
BASE_DIR = Path.cwd()
if not (BASE_DIR / "Training Dataset.arff").exists():
    BASE_DIR = Path(__file__).resolve().parent

ARFF_PATH = BASE_DIR / "Training Dataset.arff"
PREPARED_DIR = BASE_DIR / "prepared"
RUN_DIR = BASE_DIR / "runs" / "paper_reproduction"
MODEL_DIR = RUN_DIR / "models"
SEED = 3407
CV_SPLITS = 10

for path in (PREPARED_DIR, RUN_DIR, MODEL_DIR):
    path.mkdir(parents=True, exist_ok=True)

print("BASE_DIR:", BASE_DIR)
print("ARFF_PATH:", ARFF_PATH)


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def normalize_column_name(name: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", name.strip()).strip("_").lower()
    return normalized or "column"


def load_simple_arff(path: Path) -> pd.DataFrame:
    columns: list[str] = []
    rows: list[list[str]] = []
    in_data = False

    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("%"):
                continue
            lower = line.lower()
            if lower.startswith("@attribute"):
                parts = line.split()
                columns.append(parts[1])
                continue
            if lower.startswith("@data"):
                in_data = True
                continue
            if in_data:
                values = [value.strip() for value in line.split(",")]
                if len(values) == len(columns):
                    rows.append(values)

    if not columns or not rows:
        raise RuntimeError(f"Failed to parse ARFF file: {path}")

    frame = pd.DataFrame(rows, columns=[normalize_column_name(column) for column in columns])
    for column in frame.columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


# %%!
dataset = load_simple_arff(ARFF_PATH)

if "result" not in dataset.columns:
    raise RuntimeError("Expected a `result` column in the UCI ARFF file.")

result_counts = Counter(dataset["result"].astype(int).tolist())

# UCI Phishing Websites documentation uses: 1 = legitimate, -1 = phishing.
dataset["label"] = dataset["result"].astype(int).map({1: 0, -1: 1})
dataset["label_name"] = dataset["label"].map({0: "benign", 1: "phishing"})

feature_columns = [column for column in dataset.columns if column not in {"result", "label", "label_name"}]
prepared = dataset[feature_columns + ["result", "label", "label_name"]].copy()
prepared.to_csv(PREPARED_DIR / "uci_phishing_prepared.csv", index=False)

summary = {
    "rows": int(len(prepared)),
    "feature_count": int(len(feature_columns)),
    "raw_result_counts": {str(key): int(value) for key, value in result_counts.items()},
    "label_distribution": prepared["label_name"].value_counts().to_dict(),
}
write_json(RUN_DIR / "data_summary.json", summary)
print(json.dumps(summary, indent=2))

# %%!
X = prepared[feature_columns].copy()
y = prepared["label"].astype(int).copy()

ensemble = VotingClassifier(
    estimators=[
        ("dt", DecisionTreeClassifier(random_state=SEED)),
        ("rf", RandomForestClassifier(random_state=SEED, n_estimators=300, n_jobs=-1, class_weight="balanced")),
        ("knn", KNeighborsClassifier()),
    ],
    voting="hard",
    n_jobs=-1,
)

pipeline = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("scaler", StandardScaler()),
        ("selector", SelectKBest(score_func=mutual_info_classif, k=20)),
        ("model", ensemble),
    ]
)

param_grid = {
    "selector__k": [10, 15, 20, 25, 30],
    "model__dt__max_depth": [None, 6, 10],
    "model__rf__n_estimators": [200, 400],
    "model__rf__max_depth": [None, 10],
    "model__knn__n_neighbors": [3, 5, 7],
}

cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=SEED)
grid = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring={"accuracy": "accuracy", "f1": "f1"},
    refit="f1",
    cv=cv,
    n_jobs=-1,
    verbose=1,
    return_train_score=False,
)
grid.fit(X, y)

cv_results = pd.DataFrame(grid.cv_results_).sort_values(by="rank_test_f1")
cv_results.to_csv(RUN_DIR / "grid_search_results.csv", index=False)

best_estimator = grid.best_estimator_
oof_predictions = cross_val_predict(best_estimator, X, y, cv=cv, n_jobs=-1, method="predict")

metrics = {
    "best_params": grid.best_params_,
    "best_cv_f1": float(grid.best_score_),
    "oof_accuracy": float(accuracy_score(y, oof_predictions)),
    "oof_precision": float(precision_score(y, oof_predictions, zero_division=0)),
    "oof_recall": float(recall_score(y, oof_predictions, zero_division=0)),
    "oof_f1": float(f1_score(y, oof_predictions, zero_division=0)),
    "classification_report": classification_report(y, oof_predictions, output_dict=True, zero_division=0),
}
write_json(RUN_DIR / "best_model_summary.json", metrics)

with (MODEL_DIR / "enlem_best_estimator.pkl").open("wb") as handle:
    pickle.dump(best_estimator, handle)

print(json.dumps(metrics, indent=2))
print(cv_results[["rank_test_f1", "mean_test_f1", "mean_test_accuracy", "params"]].head(10))

