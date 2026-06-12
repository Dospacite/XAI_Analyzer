#!/usr/bin/env python3
"""Notebook source for reproducing Kapan and Gunal (2023).

The public repository for this paper contains the official train/test CSV files
but not training code. This notebook keeps the authors' split intact and
recreates their feature-group and classifier comparison locally.
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
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier

# %%!
BASE_DIR = Path.cwd()
if not (BASE_DIR / "phishing_dataset-main" / "phishing_dataset_train.csv").exists():
    BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "phishing_dataset-main"
TRAIN_PATH = DATA_DIR / "phishing_dataset_train.csv"
TEST_PATH = DATA_DIR / "phishing_dataset_test.csv"
PREPARED_DIR = BASE_DIR / "prepared"
RUN_DIR = BASE_DIR / "runs" / "paper_reproduction"
MODEL_DIR = RUN_DIR / "models"
SEED = 3407

for path in (PREPARED_DIR, RUN_DIR, MODEL_DIR):
    path.mkdir(parents=True, exist_ok=True)

print("BASE_DIR:", BASE_DIR)
print("TRAIN_PATH:", TRAIN_PATH)
print("TEST_PATH:", TEST_PATH)


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def build_models() -> dict[str, object]:
    return {
        "decision_tree": DecisionTreeClassifier(random_state=SEED, class_weight="balanced"),
        "svm_rbf": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", SVC(C=2.0, kernel="rbf")),
            ]
        ),
        "knn": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", KNeighborsClassifier(n_neighbors=7)),
            ]
        ),
        "gaussian_nb": GaussianNB(),
        "mlp": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=800, random_state=SEED)),
            ]
        ),
        "sgd_log_loss": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", SGDClassifier(loss="log_loss", max_iter=3000, tol=1e-4, random_state=SEED)),
            ]
        ),
    }


# %%!
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

train_df["split"] = "train"
test_df["split"] = "test"

combined = pd.concat([train_df, test_df], ignore_index=True)
combined.to_csv(PREPARED_DIR / "kapan_combined.csv", index=False)

URL_FEATURES = [
    "domain_similarity",
    "url_length",
    "http_protocol",
    "num_dot",
    "num_slash",
    "num_double_slash",
    "num_hyphen",
    "num_underscore",
    "num_equal",
    "num_paranthesis",
    "num_curly_bracket",
    "num_square_bracket",
    "num_less_and_greater",
    "num_tilde",
    "num_asterisk",
    "num_plus",
    "url_inc_at",
    "url_inc_ip",
]
HTML_FEATURES = [
    "num_a_href",
    "num_input",
    "num_button",
    "num_link_href",
    "num_iframe",
]
HTTP_FEATURES = [
    "response_history",
    "redirect",
]
FEATURE_SETS = {
    "url": URL_FEATURES,
    "html": HTML_FEATURES,
    "http": HTTP_FEATURES,
    "url_html": URL_FEATURES + HTML_FEATURES,
    "url_http": URL_FEATURES + HTTP_FEATURES,
    "html_http": HTML_FEATURES + HTTP_FEATURES,
    "all": URL_FEATURES + HTML_FEATURES + HTTP_FEATURES,
}

summary = {
    "train_rows": int(len(train_df)),
    "test_rows": int(len(test_df)),
    "train_distribution": train_df["class"].value_counts().sort_index().to_dict(),
    "test_distribution": test_df["class"].value_counts().sort_index().to_dict(),
    "feature_set_sizes": {name: len(columns) for name, columns in FEATURE_SETS.items()},
}
write_json(RUN_DIR / "data_summary.json", summary)
print(json.dumps(summary, indent=2))

# %%!
results: list[dict[str, object]] = []
best_model_name = None
best_feature_set = None
best_estimator = None
best_f1 = -1.0

y_train = train_df["class"].astype(int)
y_test = test_df["class"].astype(int)

for feature_set_name, columns in FEATURE_SETS.items():
    X_train = train_df[columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    X_test = test_df[columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    for model_name, model in build_models().items():
        start = perf_counter()
        model.fit(X_train, y_train)
        fit_seconds = perf_counter() - start

        start = perf_counter()
        predictions = model.predict(X_test)
        predict_seconds = perf_counter() - start

        report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
        metrics = {
            "feature_set": feature_set_name,
            "model": model_name,
            "feature_count": int(len(columns)),
            "accuracy": float(accuracy_score(y_test, predictions)),
            "precision": float(precision_score(y_test, predictions, zero_division=0)),
            "recall": float(recall_score(y_test, predictions, zero_division=0)),
            "f1": float(f1_score(y_test, predictions, zero_division=0)),
            "fit_seconds": float(fit_seconds),
            "predict_seconds": float(predict_seconds),
            "classification_report": report,
        }
        results.append(metrics)

        metric_path = RUN_DIR / f"{feature_set_name}__{model_name}_metrics.json"
        write_json(metric_path, metrics)

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_model_name = model_name
            best_feature_set = feature_set_name
            best_estimator = model

leaderboard = pd.DataFrame(results).drop(columns=["classification_report"]).sort_values(
    by=["f1", "accuracy", "predict_seconds"],
    ascending=[False, False, True],
)
leaderboard.to_csv(RUN_DIR / "leaderboard.csv", index=False)

if best_estimator is None or best_model_name is None or best_feature_set is None:
    raise RuntimeError("No estimator was trained.")

with (MODEL_DIR / f"{best_feature_set}__{best_model_name}.pkl").open("wb") as handle:
    pickle.dump(best_estimator, handle)

best_summary = next(
    item
    for item in results
    if item["feature_set"] == best_feature_set and item["model"] == best_model_name
)
write_json(
    RUN_DIR / "best_model_summary.json",
    {
        "best_feature_set": best_feature_set,
        "best_model": best_model_name,
        "best_f1": best_f1,
        "metrics": best_summary,
    },
)

print(leaderboard.head(20))
print("Best feature/model pair:", best_feature_set, best_model_name)
