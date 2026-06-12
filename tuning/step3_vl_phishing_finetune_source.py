#!/usr/bin/env python3
# %%!
# # Step3-VL-10B Text-Only Fine-Tuning for Explainable Phishing Classification
#
# This Colab H100 High-RAM notebook fine-tunes `stepfun-ai/Step3-VL-10B`
# with LoRA for strict JSON phishing verdicts over URL, title, and engineered
# URL/HTML/metadata features from `features/50k.csv`.
#
# Important design constraints:
# - The Step3-VL visual branch is not used. No images, image placeholders, or
#   pixel tensors are passed to the model.
# - Feature selection is computed after the domain-grouped split and only on
#   the training split.
# - Each target has 3 to 7 evidence items.
# - Target evidence order is shuffled per row with a deterministic seed.
# - Hidden source metadata is used only for diagnostics, never in prompts.
# - Default maximum sequence length is 16k for H100 High-RAM.
#
# Sources used while building this workflow:
# - Step3-VL model card: https://huggingface.co/stepfun-ai/Step3-VL-10B
# - Transformers multimodal chat templates:
#   https://huggingface.co/docs/transformers/main/chat_templating_multimodal
# - PEFT LoRA guide: https://huggingface.co/docs/peft/main/developer_guides/lora
# - TRL SFT docs: https://huggingface.co/docs/trl/v0.24.0/sft_trainer
# - NVIDIA H100 specs: https://www.nvidia.com/en-us/data-center/h100/

# %%!
# @title Install dependencies
import os
import subprocess
import sys

INSTALL_DEPS = True  # @param {type:"boolean"}

if INSTALL_DEPS:
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--upgrade",
            "torch",
            "torchvision",
            "transformers>=4.57.0",
            "accelerate",
            "datasets",
            "peft",
            "trl",
            "bitsandbytes",
            "pandas",
            "scikit-learn",
            "tldextract",
            "tqdm",
            "safetensors",
            "huggingface_hub",
        ]
    )

# %%!
# @title Mount Google Drive and configure paths
from pathlib import Path

os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

try:
    from google.colab import drive

    drive.mount("/content/drive")
except Exception as exc:
    print(f"Drive mount skipped or unavailable: {exc}")

PROJECT_DIR_STR = "/content/drive/MyDrive/XAI_Analyzer"  # @param {type:"string"}
DATA_CSV_STR = "features/50k.csv"  # @param {type:"string"}
ANALYSIS_DIR_STR = "features/50k_analysis"  # @param {type:"string"}
RUN_DIR_STR = "tuning/runs/step3_vl_10b_phishing_lora"  # @param {type:"string"}

PROJECT_DIR = Path(PROJECT_DIR_STR)
DATA_CSV = Path(DATA_CSV_STR)
ANALYSIS_DIR = Path(ANALYSIS_DIR_STR)
RUN_DIR = Path(RUN_DIR_STR)

if not DATA_CSV.is_absolute():
    DATA_CSV = PROJECT_DIR / DATA_CSV
if not ANALYSIS_DIR.is_absolute():
    ANALYSIS_DIR = PROJECT_DIR / ANALYSIS_DIR
if not RUN_DIR.is_absolute():
    RUN_DIR = PROJECT_DIR / RUN_DIR

REPORT_DIR = RUN_DIR / "reports"
ADAPTER_DIR = RUN_DIR / "adapter"
CHECKPOINT_DIR = RUN_DIR / "checkpoints"
for path in (RUN_DIR, REPORT_DIR, ADAPTER_DIR, CHECKPOINT_DIR):
    path.mkdir(parents=True, exist_ok=True)

print("PROJECT_DIR:", PROJECT_DIR)
print("DATA_CSV:", DATA_CSV)
print("ANALYSIS_DIR:", ANALYSIS_DIR)
print("RUN_DIR:", RUN_DIR)

# %%!
# @title Imports and global configuration
import gc
import hashlib
import json
import math
import random
import re
import shutil
import types
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor, StoppingCriteria, StoppingCriteriaList, Trainer, TrainerCallback, TrainingArguments
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.trainer_utils import get_last_checkpoint

try:
    import tldextract
except Exception as exc:
    tldextract = None
    print(f"tldextract unavailable; using urlparse fallback: {exc}")


SEED = 3407  # @param {type:"integer"}
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

MODEL_ID = "stepfun-ai/Step3-VL-10B"  # @param {type:"string"}
MODEL_REVISION = "5026053b0c2f5dfaa08fc2d149384162c3c8bca1"  # @param {type:"string"}
USE_LATEST_MODEL_REVISION = False  # @param {type:"boolean"}

MAX_SEQ_LENGTH = 16384  # @param {type:"integer"}
MAX_NEW_TOKENS = 768  # @param {type:"integer"}

MAX_RECORDS = 0  # @param {type:"integer"}
MAX_TRAIN_EXAMPLES = 0  # @param {type:"integer"}
MAX_EVAL_EXAMPLES = 1000  # @param {type:"integer"}
RUN_SMOKE_TEST = False  # @param {type:"boolean"}

TRAIN_SIZE = 0.85
VAL_SIZE = 0.05
TEST_SIZE = 0.10
MAX_VISIBLE_FEATURES = 64  # @param {type:"integer"}
FEATURE_SELECTION_SAMPLE = 50000  # @param {type:"integer"}
PERMUTATION_SAMPLE = 10000  # @param {type:"integer"}

PER_DEVICE_TRAIN_BATCH_SIZE = 1  # @param {type:"integer"}
PER_DEVICE_EVAL_BATCH_SIZE = 1  # @param {type:"integer"}
GRADIENT_ACCUMULATION_STEPS = 16  # @param {type:"integer"}
NUM_TRAIN_EPOCHS = 1.0  # @param {type:"number"}
MAX_STEPS = -1  # @param {type:"integer"}
SMOKE_MAX_STEPS = 5  # @param {type:"integer"}
LEARNING_RATE = 2e-4  # @param {type:"number"}
WARMUP_RATIO = 0.03  # @param {type:"number"}
LOGGING_STEPS = 10  # @param {type:"integer"}
EVAL_STEPS = 250  # @param {type:"integer"}
SAVE_STEPS = 250  # @param {type:"integer"}
SAVE_TOTAL_LIMIT = 1  # @param {type:"integer"}

LORA_R = 16  # @param {type:"integer"}
LORA_ALPHA = 32  # @param {type:"integer"}
LORA_DROPOUT = 0.05  # @param {type:"number"}

RUN_CLASSICAL_BENCHMARK = True  # @param {type:"boolean"}
DO_BASELINE_GENERATION = False  # @param {type:"boolean"}
BASELINE_GENERATION_EXAMPLES = 100  # @param {type:"integer"}
DO_TRAIN = True  # @param {type:"boolean"}
DO_EVAL_GENERATION = True  # @param {type:"boolean"}
PUSH_TO_HUB = False  # @param {type:"boolean"}
HUB_MODEL_ID = ""  # @param {type:"string"}

if RUN_SMOKE_TEST:
    MAX_RECORDS = MAX_RECORDS or 1000
    MAX_TRAIN_EXAMPLES = MAX_TRAIN_EXAMPLES or 500
    MAX_STEPS = SMOKE_MAX_STEPS
    MAX_EVAL_EXAMPLES = min(MAX_EVAL_EXAMPLES, 50)

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    print(f"GPU memory free/total GB: {free_bytes / 1e9:.2f}/{total_bytes / 1e9:.2f}")
print("MAX_SEQ_LENGTH:", MAX_SEQ_LENGTH)

# %%!
# @title Utilities
def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8")


def json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.bool_):
        return bool(value)
    return str(value)


def compact_text(value: Any, max_chars: int = 500) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text[:max_chars].rstrip()


def stable_int(*parts: Any) -> int:
    digest = hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def registered_domain(url: Any) -> str:
    text = str(url or "").strip()
    if not text:
        return "<missing>"
    try:
        if tldextract is not None:
            ext = tldextract.extract(text)
            value = ext.registered_domain or ext.fqdn or ext.domain
            return value or urlparse(text).hostname or text
    except Exception:
        pass
    parsed = urlparse(text if "://" in text else "http://" + text)
    return parsed.hostname or text


def coerce_numeric_frame(frame: pd.DataFrame) -> pd.DataFrame:
    numeric = frame.apply(pd.to_numeric, errors="coerce")
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    return numeric


def clean_float(value: Any) -> Any:
    if value is None:
        return None
    try:
        number = float(value)
    except Exception:
        return compact_text(value, 160)
    if math.isnan(number) or math.isinf(number):
        return None
    if number.is_integer():
        return int(number)
    return round(number, 6)


def read_optional_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)

# %%!
# @title Load data and existing diagnostic analysis
if not DATA_CSV.exists():
    raise FileNotFoundError(f"DATA_CSV not found: {DATA_CSV}")

read_kwargs = {"nrows": MAX_RECORDS} if MAX_RECORDS else {}
df = pd.read_csv(DATA_CSV, **read_kwargs)

required_columns = {"id", "label", "db", "collection", "url", "final_url", "title"}
missing = sorted(required_columns - set(df.columns))
if missing:
    raise ValueError(f"Missing required columns in {DATA_CSV}: {missing}")

allowed_labels = {"phishing", "benign"}
observed_labels = set(df["label"].dropna().astype(str).unique())
if not observed_labels <= allowed_labels:
    raise ValueError(f"Unexpected labels: {sorted(observed_labels)}")

analysis_summary_path = ANALYSIS_DIR / "analysis_summary.json"
analysis_summary = {}
if analysis_summary_path.exists():
    analysis_summary = json.loads(analysis_summary_path.read_text(encoding="utf-8"))

diagnostic_tables = {
    "existing_random_forest_importance": read_optional_csv(ANALYSIS_DIR / "tables" / "random_forest_importance.csv"),
    "existing_shap_global_importance": read_optional_csv(ANALYSIS_DIR / "shap" / "shap_global_importance.csv"),
    "existing_ebm_global_importance": read_optional_csv(ANALYSIS_DIR / "ebm" / "ebm_global_importance.csv"),
    "existing_feature_mutual_information": read_optional_csv(ANALYSIS_DIR / "tables" / "feature_mutual_information.csv"),
}

df["label"] = df["label"].astype(str)
df["source_diagnostic"] = df["db"].astype(str) + "." + df["collection"].astype(str)
df["registrable_domain"] = df["final_url"].fillna(df["url"]).map(registered_domain)

print("Rows:", len(df))
print("Columns:", len(df.columns))
print("Label counts:", dict(Counter(df["label"])))
print("Source counts:", dict(Counter(df["source_diagnostic"])))
if analysis_summary:
    print("Existing 50k_analysis source-label NMI:", analysis_summary.get("source_label_nmi"))
    print("Existing RF accuracy:", (analysis_summary.get("random_forest") or {}).get("accuracy"))
    print("Existing RF ROC AUC:", (analysis_summary.get("random_forest") or {}).get("roc_auc"))

# %%!
# @title Domain-grouped split before feature selection
def split_domain_groups(frame: pd.DataFrame, seed: int) -> pd.Series:
    domain_label = (
        frame.groupby("registrable_domain")["label"]
        .agg(lambda values: values.value_counts().index[0])
        .rename("domain_label")
    )
    domains = domain_label.index.to_numpy()
    labels = domain_label.to_numpy()

    stratify = labels if min(Counter(labels).values()) >= 2 else None
    train_val_domains, test_domains = train_test_split(
        domains,
        test_size=TEST_SIZE,
        random_state=seed,
        stratify=stratify,
    )

    train_val_labels = domain_label.loc[train_val_domains].to_numpy()
    val_fraction_of_train_val = VAL_SIZE / (TRAIN_SIZE + VAL_SIZE)
    stratify_train_val = train_val_labels if min(Counter(train_val_labels).values()) >= 2 else None
    train_domains, val_domains = train_test_split(
        train_val_domains,
        test_size=val_fraction_of_train_val,
        random_state=seed + 1,
        stratify=stratify_train_val,
    )

    split_by_domain = {domain: "train" for domain in train_domains}
    split_by_domain.update({domain: "validation" for domain in val_domains})
    split_by_domain.update({domain: "test" for domain in test_domains})
    return frame["registrable_domain"].map(split_by_domain)


df["split"] = split_domain_groups(df, SEED)
if df["split"].isna().any():
    raise RuntimeError("Some rows did not receive a split.")

split_summary = {
    split: {
        "rows": int((df["split"] == split).sum()),
        "labels": dict(Counter(df.loc[df["split"] == split, "label"])),
        "sources": dict(Counter(df.loc[df["split"] == split, "source_diagnostic"])),
        "domains": int(df.loc[df["split"] == split, "registrable_domain"].nunique()),
    }
    for split in ("train", "validation", "test")
}
print(json.dumps(split_summary, indent=2, ensure_ascii=False))

overlap = (
    set(df.loc[df["split"] == "train", "registrable_domain"])
    & set(df.loc[df["split"] == "validation", "registrable_domain"])
    | set(df.loc[df["split"] == "train", "registrable_domain"])
    & set(df.loc[df["split"] == "test", "registrable_domain"])
    | set(df.loc[df["split"] == "validation", "registrable_domain"])
    & set(df.loc[df["split"] == "test", "registrable_domain"])
)
if overlap:
    raise RuntimeError(f"Domain leakage across splits detected: {list(overlap)[:10]}")

save_json(REPORT_DIR / "dataset_summary.json", split_summary)

# %%!
# @title Train-only feature selection
HIDDEN_COLUMNS = {
    "id",
    "label",
    "db",
    "collection",
    "url",
    "final_url",
    "title",
    "source_diagnostic",
    "registrable_domain",
    "split",
}


def is_hidden_column(column: str) -> bool:
    return (
        column in HIDDEN_COLUMNS
        or column.startswith("label_info.")
        or column.lower() in {"source", "dataset", "target"}
    )


raw_feature_columns = [column for column in df.columns if not is_hidden_column(column)]
numeric_all = coerce_numeric_frame(df[raw_feature_columns])
numeric_feature_columns = [
    column
    for column in raw_feature_columns
    if numeric_all[column].notna().any()
]

train_df = df[df["split"] == "train"].copy()
train_numeric = coerce_numeric_frame(train_df[numeric_feature_columns])
train_numeric = train_numeric.fillna(train_numeric.median(numeric_only=True)).fillna(0.0)
nonzero_var_columns = train_numeric.columns[train_numeric.nunique(dropna=False) > 1].tolist()
train_numeric = train_numeric[nonzero_var_columns]

if FEATURE_SELECTION_SAMPLE and len(train_numeric) > FEATURE_SELECTION_SAMPLE:
    feature_sample = train_df.sample(FEATURE_SELECTION_SAMPLE, random_state=SEED)
    x_select = coerce_numeric_frame(feature_sample[nonzero_var_columns])
    x_select = x_select.fillna(train_numeric.median(numeric_only=True)).fillna(0.0)
    y_select = feature_sample["label"].to_numpy()
else:
    x_select = train_numeric
    y_select = train_df["label"].to_numpy()

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_select)

mi = mutual_info_classif(x_select, y_encoded, random_state=SEED, discrete_features="auto")
mi_df = pd.DataFrame({"feature": x_select.columns, "mutual_info": mi}).sort_values("mutual_info", ascending=False)

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=16,
    random_state=SEED,
    n_jobs=-1,
    class_weight="balanced_subsample",
)
rf.fit(x_select, y_encoded)
rf_df = pd.DataFrame({"feature": x_select.columns, "rf_importance": rf.feature_importances_}).sort_values(
    "rf_importance", ascending=False
)

if PERMUTATION_SAMPLE and len(x_select) > PERMUTATION_SAMPLE:
    perm_indices = np.random.default_rng(SEED).choice(len(x_select), size=PERMUTATION_SAMPLE, replace=False)
    x_perm = x_select.iloc[perm_indices]
    y_perm = y_encoded[perm_indices]
else:
    x_perm = x_select
    y_perm = y_encoded

perm = permutation_importance(rf, x_perm, y_perm, n_repeats=3, random_state=SEED, n_jobs=-1)
perm_df = pd.DataFrame(
    {
        "feature": x_select.columns,
        "permutation_importance_mean": perm.importances_mean,
        "permutation_importance_std": perm.importances_std,
    }
).sort_values("permutation_importance_mean", ascending=False)

SAFETY_ALLOWLIST = {
    "url.scheme_is_https",
    "url.hostname_length",
    "url.registrable_domain_length",
    "url.subdomain_length",
    "url.path_length",
    "url.query_length",
    "url.path_segment_count",
    "url.query_parameter_count",
    "url.subdomain_label_count",
    "url.dot_count",
    "url.hyphen_count",
    "url.digit_count",
    "url.hostname_digit_count",
    "url.digit_ratio",
    "url.token_count",
    "url.average_token_length",
    "url.longest_token_length",
    "url.host_is_ip_address",
    "url.punycode_present",
    "url.https_token_in_hostname",
    "url.path_or_query_contains_url",
    "url.character_entropy",
    "html.visible_text_length",
    "html.visible_word_count",
    "html.visible_text_to_html_ratio",
    "html.title_length",
    "html.current_domain_token_in_title",
    "html.title_url_token_overlap_ratio",
    "html.title_registered_domain_token_present",
    "html.meta_refresh_count",
    "html.total_tag_count",
    "html.unique_tag_count",
    "html.anchors_with_href_count",
    "html.null_or_empty_anchor_count",
    "html.placeholder_link_ratio",
    "html.internal_anchor_count",
    "html.external_anchor_count",
    "html.external_anchor_ratio",
    "html.external_to_internal_anchor_ratio",
    "html.form_count",
    "html.credential_form_present",
    "html.password_form_external_action_present",
    "html.password_form_null_action_present",
    "html.null_form_action_count",
    "html.external_form_action_count",
    "html.post_form_count",
    "html.input_count",
    "html.text_input_count",
    "html.password_input_count",
    "html.email_input_count",
    "html.hidden_input_count",
    "html.hidden_input_ratio",
    "html.submit_button_count",
    "html.iframe_count",
    "html.external_iframe_count",
    "html.script_tag_count",
    "html.external_script_count",
    "html.external_script_ratio",
    "html.inline_script_count",
    "html.external_stylesheet_count",
    "html.external_stylesheet_ratio",
    "html.image_count",
    "html.external_image_ratio",
    "html.unique_external_resource_domain_count",
    "html.external_resource_ratio",
    "html.hidden_element_present",
    "html.javascript_redirect_present",
    "html.eval_call_count",
    "html.atob_call_count",
    "html.document_write_count",
    "html.alert_or_popup_present",
    "html.onmouseover_handler_count",
    "html.right_click_disabling_present",
    "html.footer_present",
    "html.navigation_present",
    "html.privacy_or_terms_link_present",
    "metadata.redirect_count",
    "metadata.redirect_domain_change_count",
    "metadata.final_url_changed",
    "metadata.final_host_changed",
    "metadata.final_scheme_changed",
}


def ranked_features(table: pd.DataFrame, value_column: str, top_k: int) -> dict[str, int]:
    table = table.sort_values(value_column, ascending=False)
    return {feature: rank for rank, feature in enumerate(table["feature"].head(top_k), 1)}


rank_maps = [
    ranked_features(mi_df, "mutual_info", max(MAX_VISIBLE_FEATURES * 2, 20)),
    ranked_features(rf_df, "rf_importance", max(MAX_VISIBLE_FEATURES * 2, 20)),
    ranked_features(perm_df, "permutation_importance_mean", max(MAX_VISIBLE_FEATURES * 2, 20)),
]

feature_scores: dict[str, float] = defaultdict(float)
for rank_map in rank_maps:
    for feature, rank in rank_map.items():
        feature_scores[feature] += 1.0 / rank
for feature in SAFETY_ALLOWLIST:
    if feature in nonzero_var_columns:
        feature_scores[feature] += 0.05

selected_features = [
    feature
    for feature, _score in sorted(feature_scores.items(), key=lambda item: item[1], reverse=True)
    if feature in nonzero_var_columns
][:MAX_VISIBLE_FEATURES]

if len(selected_features) < min(16, len(nonzero_var_columns)):
    selected_features = list(mi_df["feature"].head(MAX_VISIBLE_FEATURES))

print("Selected feature count:", len(selected_features))
print("Selected features:")
for index, feature in enumerate(selected_features, 1):
    print(f"{index:02d}. {feature}")

train_feature_report = {
    "max_visible_features": MAX_VISIBLE_FEATURES,
    "selected_features": selected_features,
    "top_mutual_info": mi_df.head(50).to_dict(orient="records"),
    "top_random_forest": rf_df.head(50).to_dict(orient="records"),
    "top_permutation": perm_df.head(50).to_dict(orient="records"),
    "note": "All feature selection tables in this report were computed on the training split only.",
}
save_json(REPORT_DIR / "selected_features.json", train_feature_report)

# %%!
# @title Split-safe classical benchmark on selected features
def make_model_matrix(frame: pd.DataFrame, medians: pd.Series) -> pd.DataFrame:
    matrix = coerce_numeric_frame(frame[selected_features])
    return matrix.fillna(medians).fillna(0.0)


def binary_metrics(y_true: list[str], y_pred: list[str], phishing_prob: np.ndarray | None = None) -> dict[str, Any]:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=["benign", "phishing"],
        zero_division=0,
    )
    report = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "labels": {
            label: {
                "precision": precision[index],
                "recall": recall[index],
                "f1": f1[index],
                "support": int(support[index]),
            }
            for index, label in enumerate(["benign", "phishing"])
        },
        "confusion_matrix_labels": ["benign", "phishing"],
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=["benign", "phishing"]).tolist(),
    }
    if phishing_prob is not None:
        y_binary = np.array([1 if label == "phishing" else 0 for label in y_true])
        if len(set(y_binary)) == 2:
            report["roc_auc"] = roc_auc_score(y_binary, phishing_prob)
            report["pr_auc"] = average_precision_score(y_binary, phishing_prob)
    return report


classical_benchmark = {}
if RUN_CLASSICAL_BENCHMARK:
    train_medians = coerce_numeric_frame(train_df[selected_features]).median(numeric_only=True)
    x_train_bench = make_model_matrix(train_df, train_medians)
    y_train_bench = train_df["label"].to_numpy()
    classical_model = RandomForestClassifier(
        n_estimators=500,
        max_depth=18,
        min_samples_leaf=2,
        random_state=SEED,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    classical_model.fit(x_train_bench, y_train_bench)
    phishing_index = list(classical_model.classes_).index("phishing")
    for split in ("validation", "test"):
        split_frame = df[df["split"] == split].copy()
        x_split = make_model_matrix(split_frame, train_medians)
        y_true = split_frame["label"].tolist()
        y_pred = classical_model.predict(x_split).tolist()
        phishing_prob = classical_model.predict_proba(x_split)[:, phishing_index]
        classical_benchmark[split] = binary_metrics(y_true, y_pred, phishing_prob)
    classical_benchmark["training_rows"] = int(len(train_df))
    classical_benchmark["selected_feature_count"] = int(len(selected_features))
    classical_benchmark["selected_features"] = selected_features
    classical_benchmark["leakage_note"] = (
        "This benchmark uses only features selected on the training split; validation and test use the frozen selected feature list."
    )
    save_json(REPORT_DIR / "classical_benchmark_metrics.json", classical_benchmark)
    print(json.dumps(classical_benchmark, indent=2, ensure_ascii=False))
else:
    print("Classical benchmark skipped because RUN_CLASSICAL_BENCHMARK=False.")

# %%!
# @title Prompt and evidence construction
SYSTEM_PROMPT = (
    "You are a text-only phishing website classifier. "
    "Use only the URL, title, and provided engineered feature values. "
    "Do not use or infer from screenshots, images, visual layout, dataset source, collection, hidden labels, or training metadata. "
    "Return strict JSON only, with no Markdown and no chain-of-thought. "
    "Use this exact schema: "
    '{"verdict":"phishing|benign","confidence_level":"low|medium|high",'
    '"evidence":[{"feature":"feature.name","value":0,"direction":"suspicious|benign|neutral",'
    '"severity":"low|medium|high","statement":"short observable reason"}],'
    '"explanation":"short evidence-grounded explanation"}. '
    "The evidence array must contain at least 3 and at most 7 items. "
    "Every evidence item must cite a feature name that appears in the provided feature_values object."
)

FEATURE_LABELS = {
    "url.scheme_is_https": "HTTPS scheme flag",
    "url.hostname_length": "hostname length",
    "url.registrable_domain_length": "registered-domain length",
    "url.subdomain_length": "subdomain length",
    "url.path_length": "URL path length",
    "url.query_length": "URL query length",
    "url.path_segment_count": "path segment count",
    "url.query_parameter_count": "query parameter count",
    "url.subdomain_label_count": "subdomain label count",
    "url.dot_count": "dot count in URL",
    "url.hyphen_count": "hyphen count in URL",
    "url.digit_count": "digit count in URL",
    "url.hostname_digit_count": "digit count in hostname",
    "url.digit_ratio": "digit ratio in URL",
    "url.token_count": "URL token count",
    "url.average_token_length": "average URL token length",
    "url.longest_token_length": "longest URL token length",
    "url.host_is_ip_address": "IP-address hostname flag",
    "url.punycode_present": "punycode hostname flag",
    "url.https_token_in_hostname": "HTTPS token in hostname flag",
    "url.path_or_query_contains_url": "embedded URL in path/query flag",
    "url.character_entropy": "URL character entropy",
    "html.visible_text_length": "visible text length",
    "html.visible_word_count": "visible word count",
    "html.visible_text_to_html_ratio": "visible-text to HTML ratio",
    "html.title_length": "title length",
    "html.current_domain_token_in_title": "current-domain token in title flag",
    "html.title_url_token_overlap_ratio": "title and URL token overlap ratio",
    "html.title_registered_domain_token_present": "registered-domain token in title flag",
    "html.total_tag_count": "HTML tag count",
    "html.unique_tag_count": "unique HTML tag count",
    "html.placeholder_link_ratio": "placeholder-link ratio",
    "html.internal_anchor_count": "internal anchor count",
    "html.external_anchor_count": "external anchor count",
    "html.external_anchor_ratio": "external anchor ratio",
    "html.external_to_internal_anchor_ratio": "external-to-internal anchor ratio",
    "html.form_count": "form count",
    "html.credential_form_present": "credential-form flag",
    "html.password_form_external_action_present": "external password-form action flag",
    "html.password_form_null_action_present": "blank password-form action flag",
    "html.null_form_action_count": "blank form action count",
    "html.external_form_action_count": "external form action count",
    "html.input_count": "input count",
    "html.text_input_count": "text input count",
    "html.password_input_count": "password input count",
    "html.email_input_count": "email input count",
    "html.hidden_input_count": "hidden input count",
    "html.hidden_input_ratio": "hidden input ratio",
    "html.submit_button_count": "submit button count",
    "html.iframe_count": "iframe count",
    "html.external_iframe_count": "external iframe count",
    "html.script_tag_count": "script tag count",
    "html.external_script_count": "external script count",
    "html.external_script_ratio": "external script ratio",
    "html.inline_script_count": "inline script count",
    "html.external_stylesheet_count": "external stylesheet count",
    "html.external_stylesheet_ratio": "external stylesheet ratio",
    "html.image_count": "image count",
    "html.external_image_ratio": "external image ratio",
    "html.unique_external_resource_domain_count": "unique external resource domain count",
    "html.external_resource_ratio": "external resource ratio",
    "html.hidden_element_present": "hidden element flag",
    "html.javascript_redirect_present": "JavaScript redirect flag",
    "html.eval_call_count": "JavaScript eval call count",
    "html.atob_call_count": "JavaScript atob call count",
    "html.document_write_count": "document.write call count",
    "html.alert_or_popup_present": "alert or popup flag",
    "html.onmouseover_handler_count": "mouseover handler count",
    "html.right_click_disabling_present": "right-click disabling flag",
    "html.footer_present": "footer flag",
    "html.navigation_present": "navigation flag",
    "html.privacy_or_terms_link_present": "privacy or terms link flag",
    "metadata.redirect_count": "redirect count",
    "metadata.redirect_domain_change_count": "redirect domain-change count",
    "metadata.final_url_changed": "final URL changed flag",
    "metadata.final_host_changed": "final host changed flag",
    "metadata.final_scheme_changed": "final scheme changed flag",
}


def selected_feature_importance_percentiles() -> dict[str, float]:
    scores = np.array([feature_scores.get(feature, 0.0) for feature in selected_features], dtype=float)
    if len(scores) == 0 or np.all(scores == scores[0]):
        return {feature: 0.5 for feature in selected_features}
    order = np.argsort(np.argsort(scores))
    return {
        feature: float(order[index] / max(1, len(scores) - 1))
        for index, feature in enumerate(selected_features)
    }


def build_feature_evidence_profiles() -> dict[str, dict[str, Any]]:
    profiles = {}
    importance_percentiles = selected_feature_importance_percentiles()
    train_matrix = coerce_numeric_frame(train_df[selected_features])
    train_matrix = train_matrix.replace([np.inf, -np.inf], np.nan)
    train_matrix = train_matrix.fillna(train_matrix.median(numeric_only=True)).fillna(0.0)
    labels = train_df["label"].reset_index(drop=True)
    for feature in selected_features:
        series = train_matrix[feature].reset_index(drop=True)
        benign = series[labels == "benign"]
        phishing = series[labels == "phishing"]
        if benign.empty or phishing.empty:
            continue
        benign_mean = float(benign.mean())
        phishing_mean = float(phishing.mean())
        benign_median = float(benign.median())
        phishing_median = float(phishing.median())
        pooled_std = float(series.std(ddof=0)) or 1.0
        standardized_effect = (phishing_mean - benign_mean) / pooled_std
        absolute_effect = abs(standardized_effect)
        if absolute_effect < 0.03 and abs(phishing_median - benign_median) < 1e-9:
            continue
        suspicious_when = "high" if phishing_mean >= benign_mean else "low"
        threshold = (phishing_median + benign_median) / 2.0
        if abs(phishing_median - benign_median) < 1e-9:
            threshold = (phishing_mean + benign_mean) / 2.0
        if not math.isfinite(threshold):
            continue
        importance_pct = importance_percentiles.get(feature, 0.0)
        if absolute_effect >= 0.75 and importance_pct >= 0.65:
            severity = "high"
        elif absolute_effect >= 0.30 or importance_pct >= 0.50:
            severity = "medium"
        else:
            severity = "low"
        profiles[feature] = {
            "feature": feature,
            "label": FEATURE_LABELS.get(feature, feature),
            "suspicious_when": suspicious_when,
            "threshold": clean_float(threshold),
            "severity": severity,
            "standardized_effect": clean_float(standardized_effect),
            "absolute_effect": clean_float(absolute_effect),
            "importance_score": clean_float(feature_scores.get(feature, 0.0)),
            "importance_percentile": clean_float(importance_pct),
            "benign_mean": clean_float(benign_mean),
            "phishing_mean": clean_float(phishing_mean),
            "benign_median": clean_float(benign_median),
            "phishing_median": clean_float(phishing_median),
            "benign_q25": clean_float(benign.quantile(0.25)),
            "benign_q75": clean_float(benign.quantile(0.75)),
            "phishing_q25": clean_float(phishing.quantile(0.25)),
            "phishing_q75": clean_float(phishing.quantile(0.75)),
            "grounding": "train_split_class_statistics",
        }
    return profiles


FEATURE_EVIDENCE_PROFILES = build_feature_evidence_profiles()
if len(FEATURE_EVIDENCE_PROFILES) < 3:
    raise RuntimeError("Need at least 3 train-grounded feature evidence profiles.")

save_json(
    REPORT_DIR / "feature_evidence_profiles.json",
    {
        "note": "Evidence direction, thresholds, severity, and effect sizes are computed from the training split only.",
        "profiles": FEATURE_EVIDENCE_PROFILES,
    },
)
print("Train-grounded evidence profiles:", len(FEATURE_EVIDENCE_PROFILES))


def feature_values_for_row(row: pd.Series) -> dict[str, Any]:
    payload = {}
    for feature in selected_features:
        payload[feature] = clean_float(row.get(feature))
    return payload


def feature_group(feature: str) -> str:
    if feature.startswith("url."):
        return "url"
    if feature.startswith("html."):
        return "html"
    if feature.startswith("metadata."):
        return "metadata"
    return "other"


def make_user_payload(row: pd.Series) -> dict[str, Any]:
    feature_values = feature_values_for_row(row)
    grouped = defaultdict(dict)
    for feature, value in feature_values.items():
        grouped[feature_group(feature)][feature] = value
    return {
        "task": "Classify the website as phishing or benign and return strict JSON.",
        "input_modality": "text_and_engineered_features_only",
        "url": compact_text(row.get("url"), 1000),
        "final_url": compact_text(row.get("final_url"), 1000),
        "title": compact_text(row.get("title"), 300),
        "feature_values": feature_values,
        "feature_groups": dict(grouped),
    }


def evidence_item(feature: str, value: Any, direction: str, severity: str, statement: str) -> dict[str, Any]:
    return {
        "feature": feature,
        "value": value,
        "direction": direction,
        "severity": severity,
        "statement": statement,
    }


def analysis_grounded_statement(profile: dict[str, Any], value: Any, direction: str) -> str:
    relation = "above" if float(value) >= float(profile["threshold"]) else "below"
    association = "phishing" if profile["suspicious_when"] == "high" else "benign"
    opposite = "benign" if association == "phishing" else "phishing"
    if direction == "suspicious":
        associated_label = "phishing"
    elif direction == "benign":
        associated_label = "benign"
    else:
        associated_label = "neither class"
    return (
        f"Train-split feature analysis associates {profile['label']} with {associated_label}; "
        f"this value ({value}) is {relation} the train-derived threshold ({profile['threshold']})."
        if direction in {"suspicious", "benign"}
        else (
            f"Train-split feature analysis found {profile['label']} separates {association} from {opposite}; "
            f"this value ({value}) is near the train-derived threshold ({profile['threshold']})."
        )
    )


def candidate_evidence(row: pd.Series) -> list[dict[str, Any]]:
    values = feature_values_for_row(row)
    items = []
    for feature, value in values.items():
        if value is None:
            continue
        profile = FEATURE_EVIDENCE_PROFILES.get(feature)
        if not profile:
            continue
        try:
            numeric_value = float(value)
            threshold = float(profile["threshold"])
        except Exception:
            continue
        if profile["suspicious_when"] == "high":
            direction = "suspicious" if numeric_value >= threshold else "benign"
        else:
            direction = "suspicious" if numeric_value <= threshold else "benign"
        items.append(
            evidence_item(
                feature,
                value,
                direction,
                str(profile["severity"]),
                analysis_grounded_statement(profile, value, direction),
            )
        )
    return items


def severity_rank(severity: str) -> int:
    return {"high": 3, "medium": 2, "low": 1}.get(severity, 0)


def direction_for_label(label: str) -> str:
    return "suspicious" if label == "phishing" else "benign"


def build_target(row: pd.Series, split: str) -> dict[str, Any]:
    label = str(row["label"])
    desired = direction_for_label(label)
    items = candidate_evidence(row)
    aligned = [item for item in items if item["direction"] == desired]
    support = [item for item in items if item["direction"] != desired]

    aligned.sort(key=lambda item: (severity_rank(item["severity"]), str(item["feature"])), reverse=True)
    support.sort(key=lambda item: (severity_rank(item["severity"]), str(item["feature"])), reverse=True)

    chosen = []
    seen = set()
    for item in aligned + support:
        key = item["feature"]
        if key in seen:
            continue
        chosen.append(item)
        seen.add(key)
        if len(chosen) >= 7:
            break

    values = feature_values_for_row(row)
    if len(chosen) < 3:
        fallback_features = [
            feature
            for feature in selected_features
            if feature not in seen and values.get(feature) is not None
        ]
        fallback_features.sort()
        for feature in fallback_features:
            value = values.get(feature)
            chosen.append(
                evidence_item(
                    feature,
                    value,
                    "neutral",
                    "low",
                    f"Observed feature value for {feature}.",
                )
            )
            seen.add(feature)
            if len(chosen) >= 3:
                break

    if len(chosen) < 3:
        raise ValueError(f"Could not build at least 3 evidence items for row id={row.get('id')}")

    rng = random.Random(stable_int(SEED, row.get("id"), split))
    chosen = chosen[:7]
    rng.shuffle(chosen)

    high_count = sum(1 for item in chosen if item["severity"] == "high" and item["direction"] == desired)
    aligned_count = sum(1 for item in chosen if item["direction"] == desired)
    if high_count >= 1 and aligned_count >= 3:
        confidence = "high"
    elif aligned_count >= 2:
        confidence = "medium"
    else:
        confidence = "low"

    explanation_bits = [item["statement"] for item in chosen if item["direction"] == desired][:3]
    if not explanation_bits:
        explanation_bits = [item["statement"] for item in chosen[:3]]
    explanation = " ".join(explanation_bits)

    return {
        "verdict": label,
        "confidence_level": confidence,
        "evidence": chosen,
        "explanation": compact_text(explanation, 700),
    }


def make_messages(row: pd.Series, split: str, include_answer: bool) -> list[dict[str, Any]]:
    user_payload = make_user_payload(row)
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(user_payload, ensure_ascii=False, separators=(",", ":")),
                }
            ],
        },
    ]
    if include_answer:
        target = build_target(row, split)
        messages.append(
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(target, ensure_ascii=False, separators=(",", ":")),
                    }
                ],
            }
        )
    return messages


sample_target = build_target(train_df.iloc[0], "train")
print("Sample target:")
print(json.dumps(sample_target, indent=2, ensure_ascii=False))
if not (3 <= len(sample_target["evidence"]) <= 7):
    raise RuntimeError("Evidence count policy failed on sample target.")

# %%!
# @title Load Step3-VL processor and model
key_mapping = {
    "^vision_model": "model.vision_model",
    r"^model(?!\.(language_model|vision_model))": "model.language_model",
    "vit_large_projector": "model.vit_large_projector",
}

revision = None if USE_LATEST_MODEL_REVISION else MODEL_REVISION
processor = AutoProcessor.from_pretrained(MODEL_ID, revision=revision, trust_remote_code=True)
tokenizer = processor.tokenizer
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"


def resolve_generation_eos_token_ids(tokenizer):
    token_ids = []
    for token_id in (tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>")):
        if isinstance(token_id, int) and token_id >= 0 and token_id not in token_ids:
            token_ids.append(token_id)
    if not token_ids:
        return None
    return token_ids[0] if len(token_ids) == 1 else token_ids


GENERATION_EOS_TOKEN_ID = resolve_generation_eos_token_ids(tokenizer)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    revision=revision,
    trust_remote_code=True,
    key_mapping=key_mapping,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model.config.use_cache = False
if GENERATION_EOS_TOKEN_ID is not None and hasattr(model, "generation_config"):
    model.generation_config.eos_token_id = GENERATION_EOS_TOKEN_ID
    model.generation_config.pad_token_id = tokenizer.pad_token_id


def patch_step3_peft_embedding_accessors(model):
    """Expose standard PEFT/Trainer-compatible embedding accessors.

    Step3's inner StepRoboticsModel.get_input_embeddings is normally a
    multimodal helper that expects input_ids. Transformers gradient
    checkpointing also calls get_input_embeddings() with no arguments on
    submodules. This wrapper supports both conventions for text-only use.
    """
    base_model = model.get_base_model() if hasattr(model, "get_base_model") else model
    if not hasattr(base_model, "model") or not hasattr(base_model.model, "language_model"):
        return model
    language_model = base_model.model.language_model
    if not hasattr(language_model, "embed_tokens"):
        return model

    def peft_get_input_embeddings(*args, **kwargs):
        return language_model.embed_tokens

    def peft_set_input_embeddings(value):
        language_model.embed_tokens = value

    def step3_inner_get_input_embeddings(*args, **kwargs):
        input_ids = kwargs.get("input_ids")
        if input_ids is None and args:
            input_ids = args[0]
        if input_ids is None:
            return language_model.embed_tokens
        return language_model.embed_tokens(input_ids)

    def step3_inner_set_input_embeddings(value):
        language_model.embed_tokens = value

    model.get_input_embeddings = peft_get_input_embeddings
    model.set_input_embeddings = peft_set_input_embeddings
    base_model.get_input_embeddings = peft_get_input_embeddings
    base_model.set_input_embeddings = peft_set_input_embeddings
    base_model.model.get_input_embeddings = step3_inner_get_input_embeddings
    base_model.model.set_input_embeddings = step3_inner_set_input_embeddings
    return model


def patch_step3_prepare_inputs_for_generation(model):
    """Patch Step3 remote generation for Transformers versions with no cache_position.

    Step3's remote `prepare_inputs_for_generation` indexes `cache_position[0]`
    unconditionally. Recent Transformers can pass `cache_position=None`, which
    crashes native cached generation. This keeps the upstream parent preparation
    logic, then only checks cache_position when it exists.
    """
    base_model = model.get_base_model() if hasattr(model, "get_base_model") else model
    if not hasattr(base_model, "prepare_inputs_for_generation"):
        return model

    def patched_prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        inputs_embeds=None,
        pixel_values=None,
        attention_mask=None,
        cache_position=None,
        logits_to_keep=None,
        **kwargs,
    ):
        parent_prepare = super(type(self), self).prepare_inputs_for_generation
        model_inputs = parent_prepare(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )
        is_first_step = past_key_values is None
        if cache_position is not None:
            try:
                is_first_step = bool(cache_position[0].item() == 0)
            except Exception:
                is_first_step = bool(cache_position[0] == 0)
        if is_first_step:
            model_inputs["pixel_values"] = pixel_values
        return model_inputs

    base_model.prepare_inputs_for_generation = types.MethodType(
        patched_prepare_inputs_for_generation,
        base_model,
    )
    if hasattr(model, "base_model_prepare_inputs_for_generation"):
        model.base_model_prepare_inputs_for_generation = base_model.prepare_inputs_for_generation
    return model


def patch_step3_forward_for_cached_generation(model):
    """Return past_key_values from Step3 outer forward so generate can cache."""
    base_model = model.get_base_model() if hasattr(model, "get_base_model") else model
    if not hasattr(base_model, "model") or not hasattr(base_model, "lm_head"):
        return model

    def patched_forward(
        self,
        input_ids=None,
        num_patches=None,
        patch_pixel_values=None,
        patch_newline_mask=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        cache_position=None,
        **kwargs,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        outputs = self.model(
            input_ids=input_ids,
            num_patches=num_patches,
            patch_pixel_values=patch_pixel_values,
            patch_newline_mask=patch_newline_mask,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )
        logits = self.lm_head(outputs.last_hidden_state)
        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size)
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=getattr(outputs, "past_key_values", None),
            hidden_states=getattr(outputs, "hidden_states", None),
            attentions=getattr(outputs, "attentions", None),
        )

    base_model.forward = types.MethodType(patched_forward, base_model)
    return model


model = patch_step3_peft_embedding_accessors(model)
model = patch_step3_prepare_inputs_for_generation(model)
model = patch_step3_forward_for_cached_generation(model)

if hasattr(model, "model"):
    if hasattr(model.model, "vision_model"):
        model.model.vision_model.requires_grad_(False)
    if hasattr(model.model, "vit_large_projector"):
        model.model.vit_large_projector.requires_grad_(False)

try:
    model.gradient_checkpointing_enable()
    print("Gradient checkpointing enabled.")
except Exception as exc:
    print(f"gradient_checkpointing_enable skipped: {exc}")

print("Loaded model and processor.")

# %%!
# @title Tokenization and dataset construction
def render_chat(messages: list[dict[str, Any]], add_generation_prompt: bool = False) -> str:
    return processor.apply_chat_template(
        messages,
        add_generation_prompt=add_generation_prompt,
        tokenize=False,
    )


def render_json_generation_prompt(messages: list[dict[str, Any]]) -> str:
    # Step3's built-in generation prompt starts the assistant with "<think>".
    # This classifier evaluates final JSON only, so avoid forcing a reasoning preamble.
    return render_chat(messages, add_generation_prompt=False) + "<|im_start|>assistant\n"


def strip_visual_placeholders(text: str) -> str:
    forbidden = ["<im_patch>", "<|image_pad|>", "<|vision_start|>", "<|vision_end|>"]
    for token in forbidden:
        if token in text:
            raise ValueError(f"Visual placeholder leaked into text-only prompt: {token}")
    return text


class JsonObjectStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer, prompt_length: int):
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        generated_ids = input_ids[0, self.prompt_length :]
        if generated_ids.numel() == 0:
            return False
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        start = text.find("{")
        if start < 0:
            return False
        depth = 0
        in_string = False
        escape = False
        for char in text[start:]:
            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
            else:
                if char == '"':
                    in_string = True
                elif char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        return True
        return False


def tokenize_example(row: pd.Series, split: str) -> dict[str, Any] | None:
    full_messages = make_messages(row, split, include_answer=True)
    prompt_messages = make_messages(row, split, include_answer=False)
    full_text = strip_visual_placeholders(render_chat(full_messages, add_generation_prompt=False))
    prompt_text = strip_visual_placeholders(render_chat(prompt_messages, add_generation_prompt=True))

    full = tokenizer(full_text, truncation=True, max_length=MAX_SEQ_LENGTH, add_special_tokens=False)
    prompt = tokenizer(prompt_text, truncation=True, max_length=MAX_SEQ_LENGTH, add_special_tokens=False)
    input_ids = full["input_ids"]
    attention_mask = full["attention_mask"]
    labels = input_ids.copy()
    prompt_len = min(len(prompt["input_ids"]), len(labels))
    labels[:prompt_len] = [-100] * prompt_len
    if all(value == -100 for value in labels):
        return None
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "id": str(row.get("id") or ""),
        "label": str(row.get("label") or ""),
        "split": split,
        "prompt_text": prompt_text,
        "target_json": json.dumps(build_target(row, split), ensure_ascii=False, separators=(",", ":")),
    }


def make_split_examples(split: str) -> list[dict[str, Any]]:
    frame = df[df["split"] == split].copy()
    if split == "train" and MAX_TRAIN_EXAMPLES and len(frame) > MAX_TRAIN_EXAMPLES:
        frame = frame.sample(MAX_TRAIN_EXAMPLES, random_state=SEED)
    examples = []
    dropped = 0
    for _idx, row in tqdm(frame.iterrows(), total=len(frame), desc=f"tokenize_{split}"):
        item = tokenize_example(row, split)
        if item is None:
            dropped += 1
            continue
        examples.append(item)
    print(f"{split}: kept={len(examples)} dropped={dropped}")
    return examples


train_examples = make_split_examples("train")
validation_examples = make_split_examples("validation")
test_examples = make_split_examples("test")

if not train_examples:
    raise RuntimeError("No train examples after tokenization.")

tokenization_summary = {
    "train": len(train_examples),
    "validation": len(validation_examples),
    "test": len(test_examples),
    "max_seq_length": MAX_SEQ_LENGTH,
    "train_token_lengths": {
        "min": min(len(item["input_ids"]) for item in train_examples),
        "median": int(np.median([len(item["input_ids"]) for item in train_examples])),
        "max": max(len(item["input_ids"]) for item in train_examples),
    },
}
print(json.dumps(tokenization_summary, indent=2))
save_json(REPORT_DIR / "tokenization_summary.json", tokenization_summary)

# %%!
# @title Dataset and collator
class PhishingSFTDataset(Dataset):
    def __init__(self, examples: list[dict[str, Any]]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.examples[index]
        return {
            "input_ids": item["input_ids"],
            "attention_mask": item["attention_mask"],
            "labels": item["labels"],
        }


@dataclass
class CausalLMCollator:
    pad_token_id: int
    label_pad_token_id: int = -100
    pad_to_multiple_of: int | None = 8

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        max_len = max(len(feature["input_ids"]) for feature in features)
        if self.pad_to_multiple_of:
            max_len = int(math.ceil(max_len / self.pad_to_multiple_of) * self.pad_to_multiple_of)
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for feature in features:
            length = len(feature["input_ids"])
            pad = max_len - length
            batch["input_ids"].append(feature["input_ids"] + [self.pad_token_id] * pad)
            batch["attention_mask"].append(feature["attention_mask"] + [0] * pad)
            batch["labels"].append(feature["labels"] + [self.label_pad_token_id] * pad)
        tensor_batch = {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}
        forbidden = {"pixel_values", "patch_pixel_values", "num_patches", "patch_newline_mask", "images"}
        leaked = forbidden & set(tensor_batch)
        if leaked:
            raise RuntimeError(f"Visual inputs leaked into batch: {sorted(leaked)}")
        return tensor_batch


train_dataset = PhishingSFTDataset(train_examples)
validation_dataset = PhishingSFTDataset(validation_examples)
test_dataset = PhishingSFTDataset(test_examples)
data_collator = CausalLMCollator(pad_token_id=tokenizer.pad_token_id)

batch = data_collator([train_dataset[0]])
print({key: tuple(value.shape) for key, value in batch.items()})
if any(key in batch for key in ("pixel_values", "images")):
    raise RuntimeError("Text-only invariant failed: image tensors are present.")

# %%!
# @title Apply LoRA to text decoder modules only
target_module_names = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]
target_modules = r".*language_model.*(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"


model = patch_step3_peft_embedding_accessors(model)

lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=target_modules,
)

model = get_peft_model(model, lora_config)
model = patch_step3_peft_embedding_accessors(model)
model = patch_step3_prepare_inputs_for_generation(model)
model = patch_step3_forward_for_cached_generation(model)
if GENERATION_EOS_TOKEN_ID is not None and hasattr(model, "generation_config"):
    model.generation_config.eos_token_id = GENERATION_EOS_TOKEN_ID
    model.generation_config.pad_token_id = tokenizer.pad_token_id
model.print_trainable_parameters()

trainable_names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
bad_trainable = [
    name
    for name in trainable_names
    if any(part in name for part in ("vision_model", "vit_large_projector"))
]
if bad_trainable:
    raise RuntimeError(f"Visual parameters are trainable, but should be frozen: {bad_trainable[:10]}")

# %%!
# @title Custom shifted-loss trainer and memory probe
def get_step3_base_model(model):
    return model.get_base_model() if hasattr(model, "get_base_model") else model


def get_step3_language_model(model):
    base_model = get_step3_base_model(model)
    if not hasattr(base_model, "model") or not hasattr(base_model.model, "language_model"):
        raise RuntimeError("Could not locate Step3 language_model for text-only embeddings.")
    return base_model.model.language_model


def text_only_forward(model, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs):
    """Run Step3 with precomputed text embeddings and no multimodal merge path."""
    language_model = get_step3_language_model(model)
    inputs_embeds = language_model.embed_tokens(input_ids)
    return model(
        input_ids=None,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        **kwargs,
    )


class Step3TextOnlyTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        input_ids = inputs.pop("input_ids")
        attention_mask = inputs.pop("attention_mask")
        outputs = text_only_forward(model, input_ids=input_ids, attention_mask=attention_mask, **inputs)
        logits = outputs.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )
        return (loss, outputs) if return_outputs else loss


def memory_probe() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    probe_loader = DataLoader(train_dataset, batch_size=1, collate_fn=data_collator)
    probe_batch = next(iter(probe_loader))
    probe_batch = {key: value.to(model.device) for key, value in probe_batch.items()}
    model.train()
    labels = probe_batch.pop("labels")
    input_ids = probe_batch.pop("input_ids")
    attention_mask = probe_batch.pop("attention_mask")
    outputs = text_only_forward(model, input_ids=input_ids, attention_mask=attention_mask, **probe_batch)
    logits = outputs.logits
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )
    if not torch.isfinite(loss):
        raise RuntimeError(f"Non-finite first-batch loss: {loss}")
    loss.backward()
    model.zero_grad(set_to_none=True)
    if torch.cuda.is_available():
        print("Peak allocated GB after probe:", torch.cuda.max_memory_allocated() / 1e9)
        torch.cuda.empty_cache()
    print("First-batch loss:", float(loss.detach().cpu()))


memory_probe()

# %%!
# @title Optional pre-training Step3 generation benchmark
def pretrain_extract_json_object(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    try:
        return json.loads(cleaned)
    except Exception:
        pass
    start = cleaned.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(cleaned)):
        char = cleaned[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
        else:
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(cleaned[start : index + 1])
                    except Exception:
                        return None
    return None


def pretrain_normalize_verdict(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"phishing", "benign"}:
        return text
    if "phish" in text:
        return "phishing"
    if "benign" in text or "legitimate" in text or "safe" in text:
        return "benign"
    return "unparseable"


def pretrain_validate_payload(payload: Any, prompt_features: set[str]) -> dict[str, Any]:
    result = {
        "json_parseable": isinstance(payload, dict),
        "schema_valid": False,
        "verdict": "unparseable",
        "evidence_count": 0,
        "evidence_count_valid": False,
        "grounded_evidence": 0,
        "duplicate_evidence": 0,
    }
    if not isinstance(payload, dict):
        return result
    verdict = pretrain_normalize_verdict(payload.get("verdict"))
    evidence = payload.get("evidence") if isinstance(payload.get("evidence"), list) else []
    features = [str(item.get("feature") or "") for item in evidence if isinstance(item, dict)]
    grounded = sum(1 for feature in features if feature in prompt_features)
    duplicates = len(features) - len(set(features))
    result.update(
        {
            "schema_valid": verdict in {"phishing", "benign"}
            and payload.get("confidence_level") in {"low", "medium", "high"}
            and 3 <= len(evidence) <= 7
            and grounded == len(evidence)
            and duplicates == 0,
            "verdict": verdict,
            "evidence_count": len(evidence),
            "evidence_count_valid": 3 <= len(evidence) <= 7,
            "grounded_evidence": grounded,
            "duplicate_evidence": duplicates,
        }
    )
    return result


def generate_for_row_pretraining(row: pd.Series) -> dict[str, Any]:
    messages = make_messages(row, str(row["split"]), include_answer=False)
    prompt_text = strip_visual_placeholders(render_json_generation_prompt(messages))
    encoded = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LENGTH, add_special_tokens=False)
    encoded = {key: value.to(model.device) for key, value in encoded.items()}
    prompt_length = encoded["input_ids"].shape[-1]
    with torch.no_grad():
        generated = model.generate(
            **encoded,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            use_cache=True,
            eos_token_id=GENERATION_EOS_TOKEN_ID,
            pad_token_id=tokenizer.pad_token_id,
            stopping_criteria=StoppingCriteriaList([JsonObjectStoppingCriteria(tokenizer, prompt_length)]),
        )
    decoded = tokenizer.decode(generated[0, prompt_length:], skip_special_tokens=True)
    payload = pretrain_extract_json_object(decoded)
    validation = pretrain_validate_payload(payload, set(feature_values_for_row(row).keys()))
    return {
        "id": str(row.get("id") or ""),
        "split": str(row.get("split") or ""),
        "true_label": str(row.get("label") or ""),
        "raw_generation": decoded,
        "parsed": payload,
        **validation,
    }


def summarize_generation_rows(rows: list[dict[str, Any]], split: str, stage: str) -> dict[str, Any]:
    valid_predictions = [row for row in rows if row["verdict"] in {"phishing", "benign"}]
    metrics = {
        "stage": stage,
        "split": split,
        "count": len(rows),
        "json_parse_rate": sum(row["json_parseable"] for row in rows) / max(1, len(rows)),
        "schema_valid_rate": sum(row["schema_valid"] for row in rows) / max(1, len(rows)),
        "evidence_count_valid_rate": sum(row["evidence_count_valid"] for row in rows) / max(1, len(rows)),
        "grounded_evidence_rate": (
            sum(row["grounded_evidence"] for row in rows) / max(1, sum(row["evidence_count"] for row in rows))
        ),
        "duplicate_evidence_total": sum(row["duplicate_evidence"] for row in rows),
    }
    if valid_predictions:
        y_true = [row["true_label"] for row in valid_predictions]
        y_pred = [row["verdict"] for row in valid_predictions]
        metrics.update(binary_metrics(y_true, y_pred))
    return metrics


if DO_BASELINE_GENERATION:
    baseline_rows = []
    baseline_frame = df[df["split"] == "validation"].copy()
    if BASELINE_GENERATION_EXAMPLES and len(baseline_frame) > BASELINE_GENERATION_EXAMPLES:
        baseline_frame = baseline_frame.sample(BASELINE_GENERATION_EXAMPLES, random_state=SEED)
    model.eval()
    for _idx, row in tqdm(baseline_frame.iterrows(), total=len(baseline_frame), desc="generate_pretrain_validation"):
        baseline_rows.append(generate_for_row_pretraining(row))
    baseline_metrics = summarize_generation_rows(baseline_rows, "validation", "pretrain_step3")
    save_json(REPORT_DIR / "pretrain_step3_generation_benchmark.json", baseline_metrics)
    with (REPORT_DIR / "pretrain_step3_generation_predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in baseline_rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=json_default) + "\n")
    print(json.dumps(baseline_metrics, indent=2, ensure_ascii=False))
else:
    print("Pre-training Step3 generation benchmark skipped because DO_BASELINE_GENERATION=False.")

# %%!
# @title Training
training_config = {
    "model_id": MODEL_ID,
    "model_revision": revision or "latest",
    "max_seq_length": MAX_SEQ_LENGTH,
    "per_device_train_batch_size": PER_DEVICE_TRAIN_BATCH_SIZE,
    "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
    "num_train_epochs": NUM_TRAIN_EPOCHS,
    "max_steps": MAX_STEPS,
    "learning_rate": LEARNING_RATE,
    "warmup_ratio": WARMUP_RATIO,
    "lora_r": LORA_R,
    "lora_alpha": LORA_ALPHA,
    "lora_dropout": LORA_DROPOUT,
    "target_modules": target_module_names,
    "target_modules_regex": target_modules,
    "visual_branch_used": False,
}
save_json(REPORT_DIR / "training_config.json", training_config)


class KeepLatestCheckpointCallback(TrainerCallback):
    def __init__(self, checkpoint_dir: Path, keep: int = 1):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.keep = max(1, int(keep))

    def _checkpoint_step(self, path: Path) -> int:
        match = re.search(r"checkpoint-(\d+)$", path.name)
        return int(match.group(1)) if match else -1

    def _cleanup(self) -> None:
        checkpoints = [
            path
            for path in self.checkpoint_dir.glob("checkpoint-*")
            if path.is_dir() and self._checkpoint_step(path) >= 0
        ]
        checkpoints.sort(key=self._checkpoint_step, reverse=True)
        for old_checkpoint in checkpoints[self.keep :]:
            shutil.rmtree(old_checkpoint, ignore_errors=True)

    def on_save(self, args, state, control, **kwargs):
        self._cleanup()
        return control

    def on_train_end(self, args, state, control, **kwargs):
        self._cleanup()
        return control


training_args = TrainingArguments(
    output_dir=str(CHECKPOINT_DIR),
    per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    num_train_epochs=NUM_TRAIN_EPOCHS,
    max_steps=MAX_STEPS,
    learning_rate=LEARNING_RATE,
    warmup_ratio=WARMUP_RATIO,
    logging_steps=LOGGING_STEPS,
    eval_strategy="steps" if validation_examples else "no",
    eval_steps=EVAL_STEPS,
    save_steps=SAVE_STEPS,
    save_total_limit=SAVE_TOTAL_LIMIT,
    bf16=True,
    fp16=False,
    gradient_checkpointing=True,
    optim="adamw_torch",
    max_grad_norm=0.3,
    report_to=[],
    remove_unused_columns=False,
)

trainer = Step3TextOnlyTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset if validation_examples else None,
    data_collator=data_collator,
    callbacks=[KeepLatestCheckpointCallback(CHECKPOINT_DIR, SAVE_TOTAL_LIMIT)],
)

if DO_TRAIN:
    last_checkpoint = get_last_checkpoint(str(CHECKPOINT_DIR)) if CHECKPOINT_DIR.exists() else None
    if last_checkpoint:
        print("Resuming from latest checkpoint:", last_checkpoint)
    else:
        print("No checkpoint found; starting fresh.")
    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
    print(train_result)
    trainer.save_model(str(ADAPTER_DIR))
    processor.save_pretrained(str(ADAPTER_DIR))
else:
    last_checkpoint = get_last_checkpoint(str(CHECKPOINT_DIR)) if CHECKPOINT_DIR.exists() else None
    if last_checkpoint:
        print("Training skipped because DO_TRAIN=False.")
        print("Loading latest checkpoint for evaluation:", last_checkpoint)
        trainer._load_from_checkpoint(last_checkpoint, model)
        model = trainer.model
        model = patch_step3_peft_embedding_accessors(model)
        model = patch_step3_prepare_inputs_for_generation(model)
        model = patch_step3_forward_for_cached_generation(model)
        trainer.model = model
    else:
        raise FileNotFoundError(
            f"DO_TRAIN=False but no checkpoint was found in {CHECKPOINT_DIR}. "
            "Set DO_TRAIN=True to train or make sure the latest checkpoint is present in Drive."
        )

# %%!
# @title Generation and robust JSON parsing
def extract_json_object(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    try:
        return json.loads(cleaned)
    except Exception:
        pass
    start = cleaned.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(cleaned)):
        char = cleaned[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
        else:
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(cleaned[start : index + 1])
                    except Exception:
                        return None
    return None


def normalize_verdict(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"phishing", "benign"}:
        return text
    if "phish" in text:
        return "phishing"
    if "benign" in text or "legitimate" in text or "safe" in text:
        return "benign"
    return "unparseable"


def validate_payload(payload: Any, prompt_features: set[str]) -> dict[str, Any]:
    result = {
        "json_parseable": isinstance(payload, dict),
        "schema_valid": False,
        "verdict": "unparseable",
        "evidence_count": 0,
        "evidence_count_valid": False,
        "grounded_evidence": 0,
        "duplicate_evidence": 0,
    }
    if not isinstance(payload, dict):
        return result
    verdict = normalize_verdict(payload.get("verdict"))
    evidence = payload.get("evidence") if isinstance(payload.get("evidence"), list) else []
    features = [str(item.get("feature") or "") for item in evidence if isinstance(item, dict)]
    grounded = sum(1 for feature in features if feature in prompt_features)
    duplicates = len(features) - len(set(features))
    result.update(
        {
            "schema_valid": verdict in {"phishing", "benign"}
            and payload.get("confidence_level") in {"low", "medium", "high"}
            and 3 <= len(evidence) <= 7
            and grounded == len(evidence)
            and duplicates == 0,
            "verdict": verdict,
            "evidence_count": len(evidence),
            "evidence_count_valid": 3 <= len(evidence) <= 7,
            "grounded_evidence": grounded,
            "duplicate_evidence": duplicates,
        }
    )
    return result


def generate_for_row(row: pd.Series) -> dict[str, Any]:
    messages = make_messages(row, str(row["split"]), include_answer=False)
    prompt_text = strip_visual_placeholders(render_json_generation_prompt(messages))
    encoded = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LENGTH, add_special_tokens=False)
    encoded = {key: value.to(model.device) for key, value in encoded.items()}
    prompt_length = encoded["input_ids"].shape[-1]
    with torch.no_grad():
        generated = model.generate(
            **encoded,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            use_cache=True,
            eos_token_id=GENERATION_EOS_TOKEN_ID,
            pad_token_id=tokenizer.pad_token_id,
            stopping_criteria=StoppingCriteriaList([JsonObjectStoppingCriteria(tokenizer, prompt_length)]),
        )
    decoded = tokenizer.decode(generated[0, prompt_length:], skip_special_tokens=True)
    payload = extract_json_object(decoded)
    prompt_features = set(feature_values_for_row(row).keys())
    validation = validate_payload(payload, prompt_features)
    return {
        "id": str(row.get("id") or ""),
        "split": str(row.get("split") or ""),
        "true_label": str(row.get("label") or ""),
        "raw_generation": decoded,
        "parsed": payload,
        **validation,
    }


def evaluate_generation(split: str, max_examples: int) -> dict[str, Any]:
    frame = df[df["split"] == split].copy()
    if max_examples and len(frame) > max_examples:
        frame = frame.sample(max_examples, random_state=SEED)
    rows = []
    model.eval()
    for _idx, row in tqdm(frame.iterrows(), total=len(frame), desc=f"generate_{split}"):
        rows.append(generate_for_row(row))
    valid_predictions = [row for row in rows if row["verdict"] in {"phishing", "benign"}]
    metrics = {
        "split": split,
        "count": len(rows),
        "json_parse_rate": sum(row["json_parseable"] for row in rows) / max(1, len(rows)),
        "schema_valid_rate": sum(row["schema_valid"] for row in rows) / max(1, len(rows)),
        "evidence_count_valid_rate": sum(row["evidence_count_valid"] for row in rows) / max(1, len(rows)),
        "grounded_evidence_rate": (
            sum(row["grounded_evidence"] for row in rows) / max(1, sum(row["evidence_count"] for row in rows))
        ),
        "duplicate_evidence_total": sum(row["duplicate_evidence"] for row in rows),
    }
    if valid_predictions:
        y_true = [row["true_label"] for row in valid_predictions]
        y_pred = [row["verdict"] for row in valid_predictions]
        metrics["classification_report"] = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=["benign", "phishing"]).tolist()
    return {"metrics": metrics, "predictions": rows}


if DO_EVAL_GENERATION:
    all_eval = {}
    all_predictions = []
    for split in ("validation", "test"):
        result = evaluate_generation(split, MAX_EVAL_EXAMPLES)
        all_eval[split] = result["metrics"]
        all_predictions.extend(result["predictions"])
    save_json(REPORT_DIR / "metrics.json", all_eval)
    with (REPORT_DIR / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in all_predictions:
            handle.write(json.dumps(row, ensure_ascii=False, default=json_default) + "\n")
    evidence_report = {
        split: {
            "evidence_count_valid_rate": metrics["evidence_count_valid_rate"],
            "grounded_evidence_rate": metrics["grounded_evidence_rate"],
            "duplicate_evidence_total": metrics["duplicate_evidence_total"],
        }
        for split, metrics in all_eval.items()
    }
    save_json(REPORT_DIR / "evidence_report.json", evidence_report)
    benchmark_summary = {
        "classical_random_forest": classical_benchmark if RUN_CLASSICAL_BENCHMARK else None,
        "pretrain_step3_generation": (
            json.loads((REPORT_DIR / "pretrain_step3_generation_benchmark.json").read_text(encoding="utf-8"))
            if (REPORT_DIR / "pretrain_step3_generation_benchmark.json").exists()
            else None
        ),
        "posttrain_step3_generation": all_eval,
        "comparison_notes": [
            "Classical benchmark uses the same train-only selected feature list as the SFT prompts.",
            "Pre-training Step3 generation benchmark is optional because it is expensive.",
            "Post-training generation metrics include JSON validity, schema validity, evidence count compliance, evidence grounding, and classification metrics where predictions are parseable.",
        ],
    }
    save_json(REPORT_DIR / "benchmark_summary.json", benchmark_summary)
    print(json.dumps(all_eval, indent=2, ensure_ascii=False))
else:
    print("Generation evaluation skipped because DO_EVAL_GENERATION=False.")

# %%!
# @title Optional Hub push
if PUSH_TO_HUB:
    if not HUB_MODEL_ID:
        raise ValueError("Set HUB_MODEL_ID before enabling PUSH_TO_HUB.")
    model.push_to_hub(HUB_MODEL_ID)
    processor.push_to_hub(HUB_MODEL_ID)
else:
    print("Hub push skipped.")

# %%!
# @title Final artifact locations
print("Adapter directory:", ADAPTER_DIR)
print("Reports directory:", REPORT_DIR)
print("Selected features:", REPORT_DIR / "selected_features.json")
print("Classical benchmark:", REPORT_DIR / "classical_benchmark_metrics.json")
print("Benchmark summary:", REPORT_DIR / "benchmark_summary.json")
print("Metrics:", REPORT_DIR / "metrics.json")
print("Predictions:", REPORT_DIR / "predictions.jsonl")

example_prediction = None
if "all_predictions" in globals() and all_predictions:
    example_prediction = all_predictions[0]
elif (REPORT_DIR / "predictions.jsonl").exists():
    with (REPORT_DIR / "predictions.jsonl").open("r", encoding="utf-8") as handle:
        first_line = handle.readline().strip()
    if first_line:
        example_prediction = json.loads(first_line)

if example_prediction is None:
    example_output = {
        "verdict": "phishing",
        "confidence_level": "high",
        "evidence": [
            {
                "feature": "url_has_ip_address",
                "value": 1,
                "direction": "supports_phishing",
                "explanation": "The URL uses an IP address instead of a registered domain.",
            },
            {
                "feature": "brand_in_subdomain",
                "value": "paypal",
                "direction": "supports_phishing",
                "explanation": "A brand term appears in a subdomain position where impersonation is common.",
            },
            {
                "feature": "domain_age_days",
                "value": 2,
                "direction": "supports_phishing",
                "explanation": "The domain is newly registered, which is common in disposable phishing infrastructure.",
            },
        ],
    }
else:
    example_output = {
        "id": example_prediction.get("id"),
        "split": example_prediction.get("split"),
        "true_label": example_prediction.get("true_label"),
        "parsed_model_output": example_prediction.get("parsed"),
        "raw_generation": example_prediction.get("raw_generation"),
    }

print("Example output:")
print(json.dumps(example_output, indent=2, ensure_ascii=False, default=json_default))
