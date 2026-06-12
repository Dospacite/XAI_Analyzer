#!/usr/bin/env python3
"""Notebook source for reproducing Hannousse and Yahiouche (2021).

This notebook trains classical ML models on the authors' published
`dataset_B_05_2020.csv` feature matrix. The official extraction scripts shipped
with the dataset remain in this folder for optional raw re-extraction from the
dataset_A pickles, but the paper's own prepared feature table is the most
direct local-ready path for model reproduction.
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
from pathlib import Path
from time import perf_counter

import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# %%!
BASE_DIR = Path.cwd()
if not (BASE_DIR / "dataset_B_05_2020.csv").exists():
    BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "dataset_B_05_2020.csv"
PREPARED_DIR = BASE_DIR / "prepared"
RUN_DIR = BASE_DIR / "runs" / "paper_reproduction"
MODEL_DIR = RUN_DIR / "models"

SEED = 3407
TEST_SIZE = 0.20

for path in (PREPARED_DIR, RUN_DIR, MODEL_DIR):
    path.mkdir(parents=True, exist_ok=True)

print("BASE_DIR:", BASE_DIR)
print("DATA_PATH:", DATA_PATH)


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def build_models() -> dict[str, object]:
    return {
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            random_state=SEED,
            class_weight="balanced",
            n_jobs=-1,
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=400,
            random_state=SEED,
            class_weight="balanced",
            n_jobs=-1,
        ),
        "decision_tree": DecisionTreeClassifier(
            max_depth=None,
            random_state=SEED,
            class_weight="balanced",
        ),
        "logistic_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=4000, n_jobs=None)),
            ]
        ),
        "knn": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", KNeighborsClassifier(n_neighbors=7)),
            ]
        ),
        "svm_rbf": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", SVC(C=3.0, kernel="rbf", probability=True)),
            ]
        ),
    }


# %%!
dataset = pd.read_csv(DATA_PATH)
dataset = dataset.rename(columns={"status": "label"})
dataset["label"] = dataset["label"].map({"legitimate": 0, "phishing": 1})

if dataset["label"].isna().any():
    raise RuntimeError("Unexpected labels found in dataset_B_05_2020.csv")

metadata = dataset[["url", "label"]].copy()
feature_columns = [column for column in dataset.columns if column not in {"url", "label"}]
features = dataset[feature_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)

prepared = pd.concat([metadata, features], axis=1)
prepared_path = PREPARED_DIR / "dataset_B_prepared.csv"
prepared.to_csv(prepared_path, index=False)

summary = {
    "rows": int(len(prepared)),
    "feature_count": int(len(feature_columns)),
    "label_distribution": {
        "benign": int((prepared["label"] == 0).sum()),
        "phishing": int((prepared["label"] == 1).sum()),
    },
    "official_scripts_present": {
        name: (BASE_DIR / name).exists()
        for name in [
            "content_features.py",
            "url_features.py",
            "external_features.py",
            "feature_extractor.py",
        ]
    },
}

write_json(RUN_DIR / "data_summary.json", summary)

print("Prepared dataset saved to:", prepared_path)
print(json.dumps(summary, indent=2))

# %%!
X_train, X_test, y_train, y_test, meta_train, meta_test = train_test_split(
    features,
    prepared["label"],
    metadata,
    test_size=TEST_SIZE,
    random_state=SEED,
    stratify=prepared["label"],
)

results: list[dict[str, object]] = []
best_model_name = None
best_model = None
best_f1 = -1.0

for model_name, model in build_models().items():
    start = perf_counter()
    model.fit(X_train, y_train)
    fit_seconds = perf_counter() - start

    start = perf_counter()
    predictions = model.predict(X_test)
    predict_seconds = perf_counter() - start

    report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
    metrics = {
        "model": model_name,
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "fit_seconds": float(fit_seconds),
        "predict_seconds": float(predict_seconds),
        "classification_report": report,
    }
    results.append(metrics)

    with (RUN_DIR / f"{model_name}_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False)

    if metrics["f1"] > best_f1:
        best_f1 = metrics["f1"]
        best_model_name = model_name
        best_model = model

leaderboard = pd.DataFrame(results).drop(columns=["classification_report"]).sort_values(
    by=["f1", "accuracy", "recall"],
    ascending=False,
)
leaderboard.to_csv(RUN_DIR / "leaderboard.csv", index=False)

if best_model is None or best_model_name is None:
    raise RuntimeError("No model was trained.")

with (MODEL_DIR / f"{best_model_name}.pkl").open("wb") as handle:
    pickle.dump(best_model, handle)

best_summary = next(item for item in results if item["model"] == best_model_name)
write_json(
    RUN_DIR / "best_model_summary.json",
    {
        "best_model": best_model_name,
        "best_f1": best_f1,
        "test_rows": int(len(X_test)),
        "feature_count": int(X_test.shape[1]),
        "metrics": best_summary,
    },
)

print(leaderboard)
print("Best model:", best_model_name)

