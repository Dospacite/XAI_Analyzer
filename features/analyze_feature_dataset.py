#!/usr/bin/env python3
"""Analyze extracted feature JSONL for bias risk and feature importance.

Outputs include:
- dataset/bias summaries
- source-label leakage diagnostics
- model evaluation plots
- Random Forest impurity and permutation importance
- SHAP global and local explanations
- LIME local explanations
- EBM global/local explanations when `interpret` is installed
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    normalized_mutual_info_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

try:
    import shap
except Exception as exc:  # pragma: no cover - optional runtime dependency
    shap = None
    SHAP_IMPORT_ERROR = exc
else:
    SHAP_IMPORT_ERROR = None

try:
    from interpret.glassbox import ExplainableBoostingClassifier
except Exception as exc:  # pragma: no cover - optional runtime dependency
    ExplainableBoostingClassifier = None
    EBM_IMPORT_ERROR = exc
else:
    EBM_IMPORT_ERROR = None


SEED = 3407


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Feature JSONL produced by features/extract_all.py.")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/feature_analysis"))
    parser.add_argument("--max-records", type=int, default=0, help="0 means all records.")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--shap-samples", type=int, default=400)
    parser.add_argument("--lime-samples", type=int, default=8)
    parser.add_argument("--lime-features", type=int, default=12)
    parser.add_argument("--skip-shap", action="store_true")
    parser.add_argument("--skip-lime", action="store_true")
    parser.add_argument("--skip-ebm", action="store_true")
    return parser.parse_args()


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
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=json_default), encoding="utf-8")


def json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return str(value)


def safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._")
    return cleaned[:160] or "item"


def feature_frame(records: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    metadata_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    for record in records:
        db = record.get("db") or ""
        collection = record.get("collection") or ""
        source = f"{db}.{collection}" if db or collection else str(record.get("source") or "<missing>")
        label_info = record.get("label_info") or {}
        metadata_rows.append(
            {
                "id": record.get("id"),
                "label": record.get("label"),
                "source": source,
                "url": record.get("url"),
                "final_url": record.get("final_url"),
                "latest_scan": label_info.get("latest_scan"),
            }
        )
        feature_rows.append(record.get("features") or {})

    meta = pd.DataFrame(metadata_rows)
    raw_features = pd.DataFrame(feature_rows)
    numeric_features = raw_features.apply(pd.to_numeric, errors="coerce")
    categorical_cols = [
        column
        for column in raw_features.columns
        if raw_features[column].notna().any() and numeric_features[column].isna().any()
    ]
    numeric_features = numeric_features.drop(columns=categorical_cols, errors="ignore").fillna(0.0)
    if categorical_cols:
        categorical = raw_features[categorical_cols].fillna("<missing>").astype(str)
        categorical = categorical.replace({"": "<empty>"})
        categorical_features = pd.get_dummies(categorical, prefix=categorical_cols, prefix_sep="=", dtype=float)
        features = pd.concat([numeric_features, categorical_features], axis=1)
    else:
        features = numeric_features
    features = features.reindex(sorted(features.columns), axis=1)
    return meta, features


def make_dirs(root: Path) -> dict[str, Path]:
    dirs = {
        "root": root,
        "plots": root / "plots",
        "tables": root / "tables",
        "shap": root / "shap",
        "lime": root / "lime",
        "ebm": root / "ebm",
        "models": root / "models",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def save_barh(df: pd.DataFrame, label_col: str, value_col: str, title: str, output: Path) -> None:
    if df.empty:
        return
    plot_df = df.tail(30)
    plt.figure(figsize=(10, max(5, 0.28 * len(plot_df))))
    plt.barh(plot_df[label_col].astype(str), plot_df[value_col].astype(float), color="#3563a9")
    plt.title(title)
    plt.xlabel(value_col)
    plt.tight_layout()
    plt.savefig(output, dpi=180, bbox_inches="tight")
    plt.close()


def save_count_plot(series: pd.Series, title: str, output: Path) -> None:
    values = series.value_counts()
    plt.figure(figsize=(8, max(4, 0.35 * len(values))))
    plt.barh(values.index.astype(str)[::-1], values.values[::-1], color="#4f7f52")
    plt.title(title)
    plt.xlabel("count")
    plt.tight_layout()
    plt.savefig(output, dpi=180, bbox_inches="tight")
    plt.close()


def save_heatmap(table: pd.DataFrame, title: str, output: Path) -> None:
    if table.empty:
        return
    fig_width = max(7, 1.2 * len(table.columns))
    fig_height = max(4, 0.45 * len(table.index))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    matrix = table.to_numpy(dtype=float)
    im = ax.imshow(matrix, cmap="Blues", aspect="auto")
    ax.set_xticks(np.arange(len(table.columns)), labels=table.columns.astype(str))
    ax.set_yticks(np.arange(len(table.index)), labels=table.index.astype(str))
    ax.set_title(title)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, str(int(matrix[i, j])), ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def shap_values_for_positive(explainer: Any, data: pd.DataFrame, positive_index: int) -> np.ndarray:
    values = explainer.shap_values(data)
    if isinstance(values, list):
        return np.asarray(values[positive_index])
    arr = np.asarray(values)
    if arr.ndim == 3:
        if arr.shape[2] == 2:
            return arr[:, :, positive_index]
        if arr.shape[1] == 2:
            return arr[:, positive_index, :]
    return arr


def predict_positive_fn(model: Any, positive_index: int):
    def _predict(values: np.ndarray) -> np.ndarray:
        return model.predict_proba(values)[:, positive_index]

    return _predict


def lime_predict_fn(model: Any, columns: list[str]):
    def _predict(values: np.ndarray) -> np.ndarray:
        return model.predict_proba(pd.DataFrame(values, columns=columns))

    return _predict


def local_prediction_rows(model: Any, x_test: pd.DataFrame, meta_test: pd.DataFrame, classes: list[str], count: int) -> list[int]:
    probs = model.predict_proba(x_test)
    confidence = probs.max(axis=1)
    phishing_index = classes.index("phishing") if "phishing" in classes else 1
    phishing_order = np.argsort(probs[:, phishing_index])[::-1].tolist()
    uncertain_order = np.argsort(np.abs(confidence - 0.5)).tolist()
    chosen: list[int] = []
    for index in phishing_order + uncertain_order:
        if index not in chosen:
            chosen.append(index)
        if len(chosen) >= count:
            break
    return chosen


def main() -> int:
    args = parse_args()
    dirs = make_dirs(args.output_dir)

    records = list(iter_jsonl(args.input, args.max_records))
    if not records:
        raise RuntimeError(f"No JSONL records found in {args.input}")

    meta, features = feature_frame(records)
    valid_mask = meta["label"].isin(["phishing", "benign"])
    meta = meta.loc[valid_mask].reset_index(drop=True)
    features = features.loc[valid_mask].reset_index(drop=True)
    if meta["label"].nunique() != 2:
        raise RuntimeError("Need both phishing and benign labels for model-based analysis.")

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(meta["label"])
    classes = label_encoder.classes_.tolist()
    positive_index = classes.index("phishing")
    source_encoder = LabelEncoder()
    source_codes = source_encoder.fit_transform(meta["source"].astype(str))

    zero_variance = features.columns[features.nunique(dropna=False) <= 1].tolist()
    model_features = features.drop(columns=zero_variance, errors="ignore")
    if model_features.empty:
        raise RuntimeError("All features are zero-variance.")

    X_train, X_test, y_train, y_test, meta_train, meta_test = train_test_split(
        model_features,
        y,
        meta,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y,
    )

    rf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=args.seed,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    y_proba = rf.predict_proba(X_test)
    phishing_proba = y_proba[:, positive_index]

    report = classification_report(y_test, y_pred, target_names=classes, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    auc = roc_auc_score((y_test == positive_index).astype(int), phishing_proba)

    label_counts = meta["label"].value_counts()
    source_counts = meta["source"].value_counts()
    source_label = pd.crosstab(meta["source"], meta["label"])
    source_label_nmi = normalized_mutual_info_score(meta["source"], meta["label"])

    feature_summary = pd.DataFrame(
        {
            "feature": features.columns,
            "mean": features.mean().to_numpy(),
            "std": features.std(ddof=0).to_numpy(),
            "min": features.min().to_numpy(),
            "max": features.max().to_numpy(),
            "non_zero_rate": (features != 0).mean().to_numpy(),
            "unique_values": features.nunique().to_numpy(),
        }
    ).sort_values("std", ascending=False)

    from sklearn.feature_selection import mutual_info_classif

    label_mi = mutual_info_classif(model_features, y, random_state=args.seed)
    source_mi = mutual_info_classif(model_features, source_codes, random_state=args.seed)
    mi_df = pd.DataFrame(
        {
            "feature": model_features.columns,
            "label_mi": label_mi,
            "source_mi": source_mi,
            "source_to_label_mi_ratio": source_mi / np.maximum(label_mi, 1e-12),
        }
    ).sort_values(["source_mi", "label_mi"], ascending=False)

    rf_importance = pd.DataFrame(
        {"feature": model_features.columns, "importance": rf.feature_importances_}
    ).sort_values("importance", ascending=False)

    perm = permutation_importance(
        rf,
        X_test,
        y_test,
        n_repeats=8,
        random_state=args.seed,
        n_jobs=-1,
        scoring="roc_auc",
    )
    permutation_df = pd.DataFrame(
        {
            "feature": model_features.columns,
            "importance_mean": perm.importances_mean,
            "importance_std": perm.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)

    label_counts.to_csv(dirs["tables"] / "label_counts.csv")
    source_counts.to_csv(dirs["tables"] / "source_counts.csv")
    source_label.to_csv(dirs["tables"] / "source_label_crosstab.csv")
    feature_summary.to_csv(dirs["tables"] / "feature_summary.csv", index=False)
    mi_df.to_csv(dirs["tables"] / "feature_mutual_information.csv", index=False)
    rf_importance.to_csv(dirs["tables"] / "random_forest_importance.csv", index=False)
    permutation_df.to_csv(dirs["tables"] / "permutation_importance.csv", index=False)

    save_count_plot(meta["label"], "Label Distribution", dirs["plots"] / "label_distribution.png")
    save_count_plot(meta["source"], "Source Distribution", dirs["plots"] / "source_distribution.png")
    save_heatmap(source_label, "Source x Label Counts", dirs["plots"] / "source_label_crosstab.png")
    save_barh(rf_importance.head(args.top_k).iloc[::-1], "feature", "importance", "Random Forest Importance", dirs["plots"] / "random_forest_importance.png")
    save_barh(permutation_df.head(args.top_k).iloc[::-1], "feature", "importance_mean", "Permutation Importance", dirs["plots"] / "permutation_importance.png")
    save_barh(mi_df.sort_values("source_mi", ascending=False).head(args.top_k).iloc[::-1], "feature", "source_mi", "Feature Association With Source", dirs["plots"] / "feature_source_mi.png")

    ConfusionMatrixDisplay(cm, display_labels=classes).plot(values_format="d", cmap="Blues")
    plt.title("Random Forest Confusion Matrix")
    plt.tight_layout()
    plt.savefig(dirs["plots"] / "confusion_matrix.png", dpi=180, bbox_inches="tight")
    plt.close()

    RocCurveDisplay.from_predictions((y_test == positive_index).astype(int), phishing_proba)
    plt.title("Random Forest ROC Curve")
    plt.tight_layout()
    plt.savefig(dirs["plots"] / "roc_curve.png", dpi=180, bbox_inches="tight")
    plt.close()

    PrecisionRecallDisplay.from_predictions((y_test == positive_index).astype(int), phishing_proba)
    plt.title("Random Forest Precision-Recall Curve")
    plt.tight_layout()
    plt.savefig(dirs["plots"] / "precision_recall_curve.png", dpi=180, bbox_inches="tight")
    plt.close()

    joblib.dump({"model": rf, "label_encoder": label_encoder, "features": model_features.columns.tolist()}, dirs["models"] / "random_forest.joblib")

    local_indices = local_prediction_rows(rf, X_test, meta_test.reset_index(drop=True), classes, args.lime_samples)
    explanations_status: dict[str, Any] = {}

    if args.skip_shap:
        explanations_status["shap"] = "skipped by --skip-shap"
    elif shap is None:
        explanations_status["shap"] = f"skipped: {SHAP_IMPORT_ERROR}"
    else:
        sample_count = min(args.shap_samples, len(X_test))
        shap_sample = X_test.sample(sample_count, random_state=args.seed)
        explainer = shap.TreeExplainer(rf)
        shap_positive = shap_values_for_positive(explainer, shap_sample, positive_index)
        shap_importance = pd.DataFrame(
            {
                "feature": shap_sample.columns,
                "mean_abs_shap": np.abs(shap_positive).mean(axis=0),
            }
        ).sort_values("mean_abs_shap", ascending=False)
        shap_importance.to_csv(dirs["shap"] / "shap_global_importance.csv", index=False)
        save_barh(shap_importance.head(args.top_k).iloc[::-1], "feature", "mean_abs_shap", "SHAP Mean Absolute Impact", dirs["shap"] / "shap_global_importance.png")
        shap.summary_plot(shap_positive, shap_sample, show=False, max_display=args.top_k)
        plt.tight_layout()
        plt.savefig(dirs["shap"] / "shap_summary.png", dpi=180, bbox_inches="tight")
        plt.close()

        local_rows: list[dict[str, Any]] = []
        for local_index in local_indices:
            values = X_test.iloc[[local_index]]
            local_shap = shap_values_for_positive(explainer, values, positive_index)[0]
            top = np.argsort(np.abs(local_shap))[::-1][: args.lime_features]
            for rank, feature_index in enumerate(top, 1):
                local_rows.append(
                    {
                        "sample_index": int(local_index),
                        "rank": rank,
                        "id": meta_test.iloc[local_index].get("id"),
                        "true_label": classes[y_test[local_index]],
                        "predicted_label": classes[y_pred[local_index]],
                        "phishing_probability": float(phishing_proba[local_index]),
                        "feature": X_test.columns[feature_index],
                        "feature_value": float(values.iloc[0, feature_index]),
                        "shap_value": float(local_shap[feature_index]),
                    }
                )
        pd.DataFrame(local_rows).to_csv(dirs["shap"] / "shap_local_explanations.csv", index=False)
        explanations_status["shap"] = "completed"

    if args.skip_lime:
        explanations_status["lime"] = "skipped by --skip-lime"
    else:
        lime_explainer = LimeTabularExplainer(
            training_data=X_train.to_numpy(),
            feature_names=X_train.columns.tolist(),
            class_names=classes,
            mode="classification",
            discretize_continuous=True,
            random_state=args.seed,
        )
        lime_rows: list[dict[str, Any]] = []
        for local_index in local_indices:
            exp = lime_explainer.explain_instance(
                X_test.iloc[local_index].to_numpy(),
                lime_predict_fn(rf, X_train.columns.tolist()),
                num_features=args.lime_features,
                top_labels=1,
            )
            sample_id = safe_name(str(meta_test.iloc[local_index].get("id") or local_index))
            html_path = dirs["lime"] / f"lime_{local_index}_{sample_id}.html"
            exp.save_to_file(str(html_path))
            label = exp.available_labels()[0]
            for feature_text, weight in exp.as_list(label=label):
                lime_rows.append(
                    {
                        "sample_index": int(local_index),
                        "id": meta_test.iloc[local_index].get("id"),
                        "true_label": classes[y_test[local_index]],
                        "predicted_label": classes[y_pred[local_index]],
                        "phishing_probability": float(phishing_proba[local_index]),
                        "lime_label": classes[label] if label < len(classes) else str(label),
                        "feature_condition": feature_text,
                        "weight": float(weight),
                        "html": str(html_path),
                    }
                )
        pd.DataFrame(lime_rows).to_csv(dirs["lime"] / "lime_local_explanations.csv", index=False)
        explanations_status["lime"] = "completed"

    if args.skip_ebm:
        explanations_status["ebm"] = "skipped by --skip-ebm"
    elif ExplainableBoostingClassifier is None:
        explanations_status["ebm"] = f"skipped: install interpret to enable EBM ({EBM_IMPORT_ERROR})"
    else:
        ebm = ExplainableBoostingClassifier(random_state=args.seed, interactions=10)
        ebm.fit(X_train, y_train)
        ebm_pred = ebm.predict(X_test)
        ebm_proba = ebm.predict_proba(X_test)[:, positive_index]
        names = list(getattr(ebm, "term_names_", []) or X_train.columns.tolist())
        if hasattr(ebm, "term_importances"):
            scores = np.asarray(ebm.term_importances(), dtype=float)
        else:
            ebm_global = ebm.explain_global()
            scores = np.asarray(ebm_global.data().get("scores") or [], dtype=float)
        if len(names) != len(scores):
            names = names[: len(scores)]
        ebm_importance = pd.DataFrame({"feature": names, "importance": np.abs(scores)})
        ebm_importance = ebm_importance.sort_values("importance", ascending=False)
        ebm_importance.to_csv(dirs["ebm"] / "ebm_global_importance.csv", index=False)
        save_barh(ebm_importance.head(args.top_k).iloc[::-1], "feature", "importance", "EBM Global Importance", dirs["ebm"] / "ebm_global_importance.png")
        save_json(
            dirs["ebm"] / "ebm_metrics.json",
            {
                "accuracy": float(accuracy_score(y_test, ebm_pred)),
                "roc_auc": float(roc_auc_score((y_test == positive_index).astype(int), ebm_proba)),
                "classification_report": classification_report(y_test, ebm_pred, target_names=classes, output_dict=True),
            },
        )
        ebm_local = ebm.explain_local(X_test.iloc[local_indices], y_test[local_indices])
        save_json(dirs["ebm"] / "ebm_local_explanations.json", ebm_local.data())
        joblib.dump(ebm, dirs["models"] / "ebm.joblib")
        explanations_status["ebm"] = "completed"

    summary = {
        "input": str(args.input),
        "rows": int(len(meta)),
        "feature_count": int(features.shape[1]),
        "model_feature_count": int(model_features.shape[1]),
        "zero_variance_feature_count": int(len(zero_variance)),
        "zero_variance_features": zero_variance,
        "label_counts": {str(k): int(v) for k, v in label_counts.items()},
        "source_counts": {str(k): int(v) for k, v in source_counts.items()},
        "source_label_nmi": float(source_label_nmi),
        "random_forest": {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "roc_auc": float(auc),
            "classification_report": report,
        },
        "top_random_forest_features": rf_importance.head(20).to_dict("records"),
        "top_permutation_features": permutation_df.head(20).to_dict("records"),
        "top_source_associated_features": mi_df.sort_values("source_mi", ascending=False).head(20).to_dict("records"),
        "explanations": explanations_status,
    }
    save_json(dirs["root"] / "analysis_summary.json", summary)

    report_lines = [
        "# Feature Dataset Analysis",
        "",
        f"- Rows: `{len(meta)}`",
        f"- Feature columns: `{features.shape[1]}`",
        f"- Model feature columns after zero-variance removal: `{model_features.shape[1]}`",
        f"- Label counts: `{dict(label_counts)}`",
        f"- Source-label NMI: `{source_label_nmi:.4f}`",
        f"- Random Forest accuracy: `{accuracy_score(y_test, y_pred):.4f}`",
        f"- Random Forest ROC AUC: `{auc:.4f}`",
        f"- SHAP: `{explanations_status.get('shap')}`",
        f"- LIME: `{explanations_status.get('lime')}`",
        f"- EBM: `{explanations_status.get('ebm')}`",
        "",
        "## Bias Signals",
        "",
        "High `source_label_nmi` means the collection source strongly predicts the label. That is expected when phishing_db is all phishing and Tranco is all benign, but it must not be exposed as a model feature.",
        "",
        "The `feature_mutual_information.csv` table ranks features by association with both label and source. Features with high `source_mi` relative to `label_mi` deserve manual review for collection artifacts.",
        "",
        "## Key Outputs",
        "",
        "- `plots/source_label_crosstab.png`",
        "- `plots/feature_source_mi.png`",
        "- `plots/random_forest_importance.png`",
        "- `plots/permutation_importance.png`",
        "- `shap/shap_global_importance.png` and `shap/shap_summary.png`",
        "- `lime/lime_local_explanations.csv` and per-sample HTML files",
        "- `ebm/ebm_global_importance.png` when `interpret` is installed",
    ]
    (dirs["root"] / "analysis_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(f"Wrote analysis to {dirs['root']}")
    print(f"Random Forest accuracy={accuracy_score(y_test, y_pred):.4f} roc_auc={auc:.4f}")
    print(f"Source-label NMI={source_label_nmi:.4f}")
    print(f"SHAP: {explanations_status.get('shap')}")
    print(f"LIME: {explanations_status.get('lime')}")
    print(f"EBM: {explanations_status.get('ebm')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
