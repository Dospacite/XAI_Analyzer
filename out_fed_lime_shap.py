#!/usr/bin/env python3
# %% [markdown]
# # SHAP + LIME on `out_fed.jsonl`
#
# Notebook-style script with `# %%` cell markers.
#
# What this does:
# - Loads `out_fed.jsonl`
# - Expands the nested `features` object into a numeric table
# - Trains a tabular classifier on the phishing / benign labels
# - Saves evaluation outputs for the held-out test split
# - Generates local LIME explanations and global / local SHAP explanations

# %%
# Optional dependency installation.
import subprocess
import sys

INSTALL_DEPS = True

if INSTALL_DEPS:
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--upgrade",
            "joblib",
            "lime",
            "matplotlib",
            "numpy",
            "pandas",
            "scikit-learn",
            "shap",
            "tqdm",
        ]
    )

# %%
# Paths and run configuration.
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
DATA_JSONL = PROJECT_DIR / "out_fed.jsonl"
RUN_DIR = PROJECT_DIR / "runs" / "out_fed_lime_shap"
REPORT_DIR = RUN_DIR / "reports"
LIME_DIR = REPORT_DIR / "lime"
SHAP_DIR = REPORT_DIR / "shap"

for path in (RUN_DIR, REPORT_DIR, LIME_DIR, SHAP_DIR):
    path.mkdir(parents=True, exist_ok=True)

print("PROJECT_DIR:", PROJECT_DIR)
print("DATA_JSONL:", DATA_JSONL)
print("RUN_DIR:", RUN_DIR)

# %%
# Imports and global settings.
import json
import re
from collections import Counter
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm.auto import tqdm


SEED = 3407
MAX_RECORDS = 0
TEST_SIZE = 0.20
N_ESTIMATORS = 300
MAX_DEPTH = 18
MIN_SAMPLES_LEAF = 2
LIME_SAMPLE_COUNT = 8
LIME_NUM_FEATURES = 10
SHAP_SAMPLE_COUNT = 256
SHAP_TOP_K = 12
TOP_FEATURE_PLOT_COUNT = 20
TOP_SOURCE_COUNT = 15

rng = np.random.default_rng(SEED)

# %%
# JSON helpers.
def iter_jsonl(path: Path, max_records: int = 0):
    with path.open("r", encoding="utf-8") as handle:
        emitted = 0
        for line_number, line in enumerate(handle, 1):
            if max_records and emitted >= max_records:
                break
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
            emitted += 1


def save_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def safe_stem(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return cleaned[:180] or "sample"


def feature_frame_from_records(records: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    metadata_rows: list[dict[str, Any]] = []
    for record in records:
        features = record.get("features") or {}
        rows.append(features)
        metadata_rows.append(
            {
                "id": record.get("id"),
                "label": record.get("label"),
                "source": record.get("source"),
                "url": record.get("url"),
                "final_url": record.get("final_url"),
            }
        )

    feature_df = pd.DataFrame(rows)
    feature_df = feature_df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    feature_df = feature_df.reindex(sorted(feature_df.columns), axis=1)

    metadata_df = pd.DataFrame(metadata_rows)
    return metadata_df, feature_df


def top_feature_rows(feature_names: list[str], weights: np.ndarray, values: np.ndarray, top_k: int) -> list[dict[str, Any]]:
    order = np.argsort(np.abs(weights))[::-1][:top_k]
    rows: list[dict[str, Any]] = []
    for index in order:
        rows.append(
            {
                "feature": feature_names[index],
                "feature_value": float(values[index]),
                "weight": float(weights[index]),
                "abs_weight": float(abs(weights[index])),
            }
        )
    return rows


def save_barh_plot(
    labels: list[str],
    values: list[float],
    title: str,
    xlabel: str,
    output_path: Path,
    color: str = "#1f77b4",
) -> None:
    plt.figure(figsize=(10, 6))
    plt.barh(labels[::-1], values[::-1], color=color)
    plt.xlabel(xlabel)
    plt.ylabel("Category")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.show()
    plt.close()


def extract_shap_matrix(explanation: shap.Explanation) -> np.ndarray:
    values = np.asarray(explanation.values)
    if values.ndim == 2:
        return values
    if values.ndim == 3:
        if values.shape[1] == len(FEATURE_NAMES):
            return values.mean(axis=2)
        if values.shape[2] == len(FEATURE_NAMES):
            return values.mean(axis=1)
    raise ValueError(f"Unexpected SHAP values shape: {values.shape}")


def extract_shap_vector(
    explanation: shap.Explanation,
    sample_index: int,
    predicted_index: int,
) -> np.ndarray:
    values = np.asarray(explanation.values)
    if values.ndim == 2:
        return values[sample_index]
    if values.ndim == 3:
        if values.shape[1] == len(FEATURE_NAMES):
            return values[sample_index, :, predicted_index]
        if values.shape[2] == len(FEATURE_NAMES):
            return values[sample_index, predicted_index, :]
    raise ValueError(f"Unexpected SHAP values shape: {values.shape}")


# %%
# Load `out_fed.jsonl`.
records = list(iter_jsonl(DATA_JSONL, MAX_RECORDS))
if not records:
    raise RuntimeError(f"No records found in {DATA_JSONL}")

metadata_df, feature_df = feature_frame_from_records(records)
label_counts = Counter(metadata_df["label"])
FEATURE_NAMES = feature_df.columns.tolist()

print("rows:", len(metadata_df))
print("labels:", dict(label_counts))
print("feature_count:", len(feature_df.columns))

# %%
# Data diagnostics and dataset summary.
source_counts = metadata_df["source"].fillna("<missing>").value_counts()
label_distribution = metadata_df["label"].fillna("<missing>").value_counts()
source_label_crosstab = pd.crosstab(
    metadata_df["source"].fillna("<missing>"),
    metadata_df["label"].fillna("<missing>"),
)
feature_summary = pd.DataFrame(
    {
        "feature": FEATURE_NAMES,
        "mean": feature_df.mean().to_numpy(),
        "std": feature_df.std(ddof=0).to_numpy(),
        "min": feature_df.min().to_numpy(),
        "max": feature_df.max().to_numpy(),
        "non_zero_rate": ((feature_df != 0).mean()).to_numpy(),
        "unique_values": feature_df.nunique().to_numpy(),
    }
).sort_values("std", ascending=False)

feature_summary.to_csv(REPORT_DIR / "feature_summary.csv", index=False)
source_label_crosstab.to_csv(REPORT_DIR / "source_label_crosstab.csv")

data_summary = {
    "row_count": int(len(metadata_df)),
    "feature_count": int(len(FEATURE_NAMES)),
    "label_distribution": {str(key): int(value) for key, value in label_distribution.items()},
    "source_count": int(source_counts.shape[0]),
    "top_sources": {str(key): int(value) for key, value in source_counts.head(TOP_SOURCE_COUNT).items()},
    "zero_variance_features": feature_summary.loc[feature_summary["std"] == 0, "feature"].tolist(),
    "most_dense_features": feature_summary.sort_values("non_zero_rate", ascending=False)
    .head(10)[["feature", "non_zero_rate"]]
    .to_dict("records"),
}
save_json(REPORT_DIR / "data_summary.json", data_summary)

print("unique sources:", int(source_counts.shape[0]))
print("zero-variance features:", len(data_summary["zero_variance_features"]))
print("top sources:")
print(source_counts.head(TOP_SOURCE_COUNT))

save_barh_plot(
    labels=label_distribution.index.astype(str).tolist(),
    values=label_distribution.astype(float).tolist(),
    title="Label Distribution in out_fed.jsonl",
    xlabel="Row Count",
    output_path=REPORT_DIR / "label_distribution.png",
    color="#2a9d8f",
)

top_source_counts = source_counts.head(TOP_SOURCE_COUNT)
save_barh_plot(
    labels=top_source_counts.index.astype(str).tolist(),
    values=top_source_counts.astype(float).tolist(),
    title="Top Sources in out_fed.jsonl",
    xlabel="Row Count",
    output_path=REPORT_DIR / "source_distribution.png",
    color="#e76f51",
)

# %%
# Train / test split.
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(metadata_df["label"])
CLASS_NAMES = label_encoder.classes_.tolist()

X_train, X_test, y_train, y_test, meta_train, meta_test = train_test_split(
    feature_df,
    y,
    metadata_df.reset_index(drop=True),
    test_size=TEST_SIZE,
    stratify=y,
    random_state=SEED,
)

print("classes:", CLASS_NAMES)
print("train shape:", X_train.shape)
print("test shape:", X_test.shape)

# %%
# Classifier training.
classifier = RandomForestClassifier(
    n_estimators=N_ESTIMATORS,
    max_depth=MAX_DEPTH,
    min_samples_leaf=MIN_SAMPLES_LEAF,
    class_weight="balanced_subsample",
    n_jobs=-1,
    random_state=SEED,
)
classifier.fit(X_train, y_train)

bundle = {
    "model": classifier,
    "feature_names": FEATURE_NAMES,
    "class_names": CLASS_NAMES,
}
joblib.dump(bundle, RUN_DIR / "model_bundle.joblib")
print("Saved model bundle:", (RUN_DIR / "model_bundle.joblib").resolve())

# %%
# Random forest feature importance graph.
model_importance = pd.DataFrame(
    {
        "feature": FEATURE_NAMES,
        "importance": classifier.feature_importances_,
    }
).sort_values("importance", ascending=False)
model_importance.to_csv(REPORT_DIR / "feature_importance.csv", index=False)

plt.figure(figsize=(10, 6))
plt.barh(
    model_importance["feature"].head(TOP_FEATURE_PLOT_COUNT)[::-1],
    model_importance["importance"].head(TOP_FEATURE_PLOT_COUNT)[::-1],
    color="#1f77b4",
)
plt.xlabel("Random Forest Feature Importance")
plt.ylabel("Feature")
plt.title("Top Model Features on out_fed.jsonl")
plt.tight_layout()
plt.savefig(REPORT_DIR / "feature_importance.png", dpi=200, bbox_inches="tight")
plt.show()
plt.close()

print("Saved feature importance graph:", (REPORT_DIR / "feature_importance.png").resolve())

# %%
# Evaluation on the held-out set.
y_prob = classifier.predict_proba(X_test)
y_pred = np.argmax(y_prob, axis=1)

test_with_predictions = meta_test.copy()
test_with_predictions["predicted_label"] = label_encoder.inverse_transform(y_pred)
test_with_predictions["predicted_confidence"] = y_prob.max(axis=1)
test_with_predictions = test_with_predictions.reset_index(drop=True)

test_feature_frame = X_test.reset_index(drop=True)
test_with_predictions = pd.concat([test_with_predictions, test_feature_frame], axis=1)
test_with_predictions = test_with_predictions.sort_values(
    by="predicted_confidence",
    ascending=False,
).reset_index(drop=True)

accuracy = accuracy_score(y_test, y_pred)
report_text = classification_report(
    y_test,
    y_pred,
    target_names=CLASS_NAMES,
    digits=4,
)
cm = confusion_matrix(y_test, y_pred)

print(f"accuracy: {accuracy:.4f}")
print(report_text)

(REPORT_DIR / "classification_report.txt").write_text(report_text, encoding="utf-8")
pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(
    REPORT_DIR / "confusion_matrix.csv",
    index=True,
)
test_with_predictions.to_csv(REPORT_DIR / "test_predictions.csv", index=False)

misclassified = test_with_predictions.loc[
    test_with_predictions["label"] != test_with_predictions["predicted_label"]
].copy()
misclassified.to_csv(REPORT_DIR / "misclassified_predictions.csv", index=False)

confidence_by_predicted_label = (
    test_with_predictions.groupby("predicted_label", as_index=False)["predicted_confidence"]
    .mean()
    .sort_values("predicted_confidence", ascending=False)
)
confidence_by_predicted_label.to_csv(
    REPORT_DIR / "confidence_by_predicted_label.csv",
    index=False,
)

model_summary = {
    "model_type": type(classifier).__name__,
    "model_params": {
        "n_estimators": int(N_ESTIMATORS),
        "max_depth": int(MAX_DEPTH),
        "min_samples_leaf": int(MIN_SAMPLES_LEAF),
        "class_weight": "balanced_subsample",
        "random_state": int(SEED),
    },
    "dataset_split": {
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "test_size": float(TEST_SIZE),
    },
    "metrics": {
        "accuracy": float(accuracy),
        "misclassified_rows": int(len(misclassified)),
        "misclassification_rate": float(len(misclassified) / len(test_with_predictions)),
        "mean_prediction_confidence": float(test_with_predictions["predicted_confidence"].mean()),
    },
    "top_model_features": model_importance.head(10).to_dict("records"),
}
save_json(REPORT_DIR / "model_summary.json", model_summary)

print("misclassified rows:", len(misclassified))
print("mean prediction confidence:", f"{test_with_predictions['predicted_confidence'].mean():.4f}")

save_barh_plot(
    labels=confidence_by_predicted_label["predicted_label"].astype(str).tolist(),
    values=confidence_by_predicted_label["predicted_confidence"].astype(float).tolist(),
    title="Mean Confidence by Predicted Label",
    xlabel="Mean Predicted Probability",
    output_path=REPORT_DIR / "confidence_by_predicted_label.png",
    color="#457b9d",
)

# %%
# LIME explanations on high-confidence test samples.
lime_explainer = LimeTabularExplainer(
    training_data=X_train.to_numpy(dtype=float),
    feature_names=FEATURE_NAMES,
    class_names=CLASS_NAMES,
    mode="classification",
    discretize_continuous=True,
    random_state=SEED,
)

lime_records: list[dict[str, Any]] = []
lime_samples = test_with_predictions.head(LIME_SAMPLE_COUNT)

for row in tqdm(list(lime_samples.itertuples(index=False)), desc="LIME"):
    sample = np.asarray([getattr(row, feature_name) for feature_name in FEATURE_NAMES], dtype=float)
    predicted_index = CLASS_NAMES.index(row.predicted_label)
    explanation = lime_explainer.explain_instance(
        data_row=sample,
        predict_fn=classifier.predict_proba,
        num_features=LIME_NUM_FEATURES,
        labels=[predicted_index],
    )

    sample_name = safe_stem(f"{row.id}_{row.predicted_label}")
    html_path = LIME_DIR / f"{sample_name}.html"
    json_path = LIME_DIR / f"{sample_name}.json"
    explanation.save_to_file(str(html_path), labels=[predicted_index])

    payload = {
        "id": row.id,
        "source": row.source,
        "url": row.url,
        "true_label": row.label,
        "predicted_label": row.predicted_label,
        "predicted_confidence": float(row.predicted_confidence),
        "rules": [
            {"rule": rule, "weight": float(weight)}
            for rule, weight in explanation.as_list(label=predicted_index)
        ],
    }
    save_json(json_path, payload)
    lime_records.append(payload)

save_json(REPORT_DIR / "lime_summary.json", lime_records)
print("Saved LIME explanations:", LIME_DIR.resolve())

# %%
# SHAP explanations for the same classifier.
background_size = min(2000, len(X_train))
background_indices = rng.choice(len(X_train), size=background_size, replace=False)
background = X_train.iloc[background_indices]

shap_explainer = shap.Explainer(classifier, background)
shap_samples = test_with_predictions.head(min(SHAP_SAMPLE_COUNT, len(test_with_predictions))).copy()
shap_sample_frame = shap_samples[FEATURE_NAMES].copy()
shap_values = shap_explainer(shap_sample_frame)

global_matrix = extract_shap_matrix(shap_values)
global_importance = pd.DataFrame(
    {
        "feature": FEATURE_NAMES,
        "mean_abs_shap": np.abs(global_matrix).mean(axis=0),
    }
).sort_values("mean_abs_shap", ascending=False)
global_importance.to_csv(SHAP_DIR / "global_importance.csv", index=False)

plt.figure(figsize=(10, 6))
plt.barh(
    global_importance["feature"].head(20)[::-1],
    global_importance["mean_abs_shap"].head(20)[::-1],
)
plt.xlabel("Mean |SHAP value|")
plt.ylabel("Feature")
plt.title("Top SHAP Features on out_fed.jsonl")
plt.tight_layout()
plt.savefig(SHAP_DIR / "global_importance.png", dpi=200, bbox_inches="tight")
plt.show()
plt.close()

shap_records: list[dict[str, Any]] = []

for sample_index, row in enumerate(shap_samples.itertuples(index=False)):
    predicted_index = CLASS_NAMES.index(row.predicted_label)
    values = extract_shap_vector(shap_values, sample_index, predicted_index)
    feature_values = shap_sample_frame.iloc[sample_index].to_numpy(dtype=float)

    payload = {
        "id": row.id,
        "source": row.source,
        "url": row.url,
        "true_label": row.label,
        "predicted_label": row.predicted_label,
        "predicted_confidence": float(row.predicted_confidence),
        "top_features": top_feature_rows(
            feature_names=FEATURE_NAMES,
            weights=values,
            values=feature_values,
            top_k=SHAP_TOP_K,
        ),
    }
    sample_name = safe_stem(f"{row.id}_{row.predicted_label}")
    save_json(SHAP_DIR / f"{sample_name}.json", payload)
    shap_records.append(payload)

save_json(REPORT_DIR / "shap_summary.json", shap_records)
print("Saved SHAP explanations:", SHAP_DIR.resolve())

# %%
# Single-record scoring helper for ad hoc inspection.
def score_record(record: dict[str, Any]) -> dict[str, Any]:
    raw_features = record.get("features") or {}
    row = pd.DataFrame([raw_features]).reindex(columns=FEATURE_NAMES)
    row = row.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    probabilities = classifier.predict_proba(row)[0]
    predicted_index = int(np.argmax(probabilities))
    return {
        "predicted_label": CLASS_NAMES[predicted_index],
        "confidence": float(probabilities[predicted_index]),
        "probabilities": {
            class_name: float(probabilities[index])
            for index, class_name in enumerate(CLASS_NAMES)
        },
    }


example_result = score_record(records[0])
print("example prediction:", example_result)
