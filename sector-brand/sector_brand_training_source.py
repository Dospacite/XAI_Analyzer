from __future__ import annotations

# Sector and brand classifier training notebook.
# Run this notebook with the project virtual environment:
# /home/ege/Documents/Projects/XAI_Analyzer/.venv/bin/python

from pathlib import Path
import json
import os
import random
import sys

PROJECT_ROOT = Path.cwd()
if PROJECT_ROOT.name == "sector-brand":
    PROJECT_ROOT = PROJECT_ROOT.parent
elif not (PROJECT_ROOT / ".env").exists():
    for parent in Path.cwd().parents:
        if (parent / ".env").exists():
            PROJECT_ROOT = parent
            break

NOTEBOOK_DIR = PROJECT_ROOT / "sector-brand"
MODEL_DIR = NOTEBOOK_DIR / "models"
ENV_FILE = PROJECT_ROOT / ".env"

assert ENV_FILE.exists(), f"Missing .env at {ENV_FILE}"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

print("Project root:", PROJECT_ROOT)
print("Python executable:", sys.executable)
print("Model output:", MODEL_DIR)
if ".venv" not in sys.executable:
    print("Warning: this notebook is not running from the project .venv.")

# %%!
import importlib.util

required_modules = {
    "datasets": "datasets",
    "dotenv": "python-dotenv",
    "numpy": "numpy",
    "pandas": "pandas",
    "parsel": "parsel",
    "pymongo": "pymongo",
    "sklearn": "scikit-learn",
    "torch": "torch",
    "transformers": "transformers",
}

missing = [package for module, package in required_modules.items() if importlib.util.find_spec(module) is None]
if missing:
    raise RuntimeError(
        "Missing required packages in the active environment: "
        + ", ".join(missing)
        + ". Install them into the project .venv before running training."
    )

print("Dependency check passed.")

# %%!
from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from datasets import Dataset
from dotenv import load_dotenv
from pymongo import MongoClient
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

from features import common as feature_common

try:
    from IPython.display import display
except Exception:
    display = print

SEED = 3407
MODEL_NAME = "answerdotai/ModernBERT-base"

def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    return int(value)


def env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    return float(value)


# Set SMOKE_TEST=True for a fast wiring check. Set it False for full training.
SMOKE_TEST = env_bool("SECTOR_BRAND_SMOKE_TEST", False)
SMOKE_TEST_ROWS = env_int("SECTOR_BRAND_SMOKE_TEST_ROWS", 512)
MAX_ROWS = env_int("SECTOR_BRAND_MAX_ROWS", 0)

BATCH_SIZE_MONGO = env_int("SECTOR_BRAND_BATCH_SIZE_MONGO", 500)
MAX_VISIBLE_TEXT_CHARS = env_int("SECTOR_BRAND_MAX_VISIBLE_TEXT_CHARS", 6000)
MAX_HTML_CHARS = env_int("SECTOR_BRAND_MAX_HTML_CHARS", 2_000_000)
MAX_LENGTH = env_int("SECTOR_BRAND_MAX_LENGTH", 1024)

NUM_EPOCHS = env_float("SECTOR_BRAND_NUM_EPOCHS", 2)
LEARNING_RATE = env_float("SECTOR_BRAND_LEARNING_RATE", 2e-5)
WEIGHT_DECAY = env_float("SECTOR_BRAND_WEIGHT_DECAY", 0.01)
PER_DEVICE_TRAIN_BATCH_SIZE = env_int("SECTOR_BRAND_TRAIN_BATCH_SIZE", 4)
PER_DEVICE_EVAL_BATCH_SIZE = env_int("SECTOR_BRAND_EVAL_BATCH_SIZE", 8)
GRADIENT_ACCUMULATION_STEPS = env_int("SECTOR_BRAND_GRAD_ACCUM_STEPS", 8)

UNIDENTIFIED_LABEL = "unidentified"

random.seed(SEED)
np.random.seed(SEED)
set_seed(SEED)

MODEL_DIR.mkdir(parents=True, exist_ok=True)

print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# %%!
def normalize_label(value: Any) -> str:
    text = feature_common.collapse_ws(value, 200).strip()
    return text if text else UNIDENTIFIED_LABEL


def extract_visible_page_text(normalized_doc: dict[str, Any]) -> tuple[str, str]:
    html = normalized_doc.get("html") or ""
    selector = feature_common.html_selector(html)
    title = feature_common.title_from_doc_or_html(normalized_doc, selector)
    visible_text = feature_common.visible_text(selector)
    return (
        feature_common.collapse_ws(title, 300),
        feature_common.collapse_ws(visible_text, MAX_VISIBLE_TEXT_CHARS),
    )


def build_model_text(url: str, title: str, visible_text: str) -> str:
    return "\n".join(
        [
            f"URL: {feature_common.collapse_ws(url, 2000)}",
            f"TITLE: {feature_common.collapse_ws(title, 300)}",
            f"TEXT: {feature_common.collapse_ws(visible_text, MAX_VISIBLE_TEXT_CHARS)}",
        ]
    ).strip()


def content_projection() -> dict[str, int]:
    projection = feature_common.mongo_feature_projection()
    projection["url"] = 1
    projection["title"] = 1
    projection["html"] = 1
    projection["metadata.url"] = 1
    projection["metadata.final_url"] = 1
    projection["metadata.status_code"] = 1
    return projection


def iter_label_batches(collection: Any, batch_size: int):
    query = {"url": {"$exists": True, "$type": "string", "$ne": ""}}
    projection = {"_id": 0, "url": 1, "brand": 1, "sector": 1}
    batch: list[dict[str, Any]] = []
    cursor = collection.find(query, projection, batch_size=batch_size).sort("_id", 1)
    for row in cursor:
        batch.append(row)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def load_training_frame() -> pd.DataFrame:
    load_dotenv(ENV_FILE)
    mongo_uri = os.environ.get("MONGO_URI")
    if not mongo_uri:
        raise RuntimeError(f"MONGO_URI was not found in {ENV_FILE}")

    rows: list[dict[str, Any]] = []
    skipped_missing_content = 0
    skipped_empty_html = 0
    skipped_large_html = 0

    with MongoClient(mongo_uri, serverSelectionTimeoutMS=10000) as client:
        label_collection = client["phishing_db"]["phishing_urls"]
        content_collection = client["phishing_db"]["website_content"]

        for label_batch in iter_label_batches(label_collection, BATCH_SIZE_MONGO):
            by_url = {item["url"]: item for item in label_batch if item.get("url")}
            content_docs = content_collection.find(
                {"url": {"$in": list(by_url.keys())}},
                content_projection(),
                batch_size=BATCH_SIZE_MONGO,
            )

            seen_urls: set[str] = set()
            for doc in content_docs:
                url = doc.get("url") or (doc.get("metadata") or {}).get("url")
                if not url or url not in by_url:
                    continue
                seen_urls.add(url)
                normalized = feature_common.normalize_document(doc)
                html = normalized.get("html") or ""
                if not html.strip():
                    skipped_empty_html += 1
                    continue
                if MAX_HTML_CHARS and len(html) > MAX_HTML_CHARS:
                    skipped_large_html += 1
                    continue

                title, visible_text = extract_visible_page_text(normalized)
                label_row = by_url[url]
                rows.append(
                    {
                        "url": url,
                        "title": title,
                        "text": build_model_text(url, title, visible_text),
                        "brand": normalize_label(label_row.get("brand")),
                        "sector": normalize_label(label_row.get("sector")),
                    }
                )

                if SMOKE_TEST and len(rows) >= SMOKE_TEST_ROWS:
                    break
                if MAX_ROWS > 0 and len(rows) >= MAX_ROWS:
                    break

            skipped_missing_content += len(set(by_url) - seen_urls)
            if SMOKE_TEST and len(rows) >= SMOKE_TEST_ROWS:
                break
            if MAX_ROWS > 0 and len(rows) >= MAX_ROWS:
                break

    df = pd.DataFrame(rows).drop_duplicates(subset=["url", "text"]).reset_index(drop=True)
    print("Loaded rows:", len(df))
    print("Skipped missing content:", skipped_missing_content)
    print("Skipped empty html:", skipped_empty_html)
    print("Skipped large html:", skipped_large_html)
    if df.empty:
        raise RuntimeError("No training rows were loaded from MongoDB.")
    return df


df = load_training_frame()
df.head()

# %%!
def summarize_target(frame: pd.DataFrame, target: str) -> pd.DataFrame:
    counts = frame[target].value_counts().rename_axis(target).reset_index(name="count")
    print(target, "classes:", len(counts))
    print(target, "top classes:")
    display(counts.head(20))
    return counts


brand_counts = summarize_target(df, "brand")
sector_counts = summarize_target(df, "sector")

metadata = {
    "model_name": MODEL_NAME,
    "seed": SEED,
    "smoke_test": SMOKE_TEST,
    "rows": int(len(df)),
    "max_visible_text_chars": MAX_VISIBLE_TEXT_CHARS,
    "max_html_chars": MAX_HTML_CHARS,
    "max_length": MAX_LENGTH,
    "brand_class_count": int(df["brand"].nunique()),
    "sector_class_count": int(df["sector"].nunique()),
    "brand_counts": df["brand"].value_counts().to_dict(),
    "sector_counts": df["sector"].value_counts().to_dict(),
}

with (MODEL_DIR / "training_metadata.json").open("w", encoding="utf-8") as handle:
    json.dump(metadata, handle, indent=2, ensure_ascii=False)

print("Metadata written:", MODEL_DIR / "training_metadata.json")

# %%!
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def can_stratify(labels: pd.Series, test_size: float) -> bool:
    counts = labels.value_counts()
    if counts.empty or counts.min() < 2:
        return False
    requested_test_rows = int(np.ceil(len(labels) * test_size))
    return requested_test_rows >= len(counts)


def split_for_target(frame: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    stratify_first = frame[target] if can_stratify(frame[target], 0.30) else None
    train_df, temp_df = train_test_split(
        frame,
        test_size=0.30,
        random_state=SEED,
        stratify=stratify_first,
    )
    stratify_second = temp_df[target] if can_stratify(temp_df[target], 0.50) else None
    valid_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=SEED,
        stratify=stratify_second,
    )
    return (
        train_df.reset_index(drop=True),
        valid_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def make_label_maps(labels: pd.Series) -> tuple[dict[str, int], dict[int, str]]:
    label_names = sorted(str(label) for label in labels.unique())
    if UNIDENTIFIED_LABEL in label_names:
        label_names.remove(UNIDENTIFIED_LABEL)
        label_names = [UNIDENTIFIED_LABEL] + label_names
    label2id = {label: idx for idx, label in enumerate(label_names)}
    id2label = {idx: label for label, idx in label2id.items()}
    return label2id, id2label


def build_dataset(frame: pd.DataFrame, target: str, label2id: dict[str, int]) -> Dataset:
    dataset = Dataset.from_pandas(
        pd.DataFrame(
            {
                "text": frame["text"].astype(str),
                "labels": frame[target].map(label2id).astype(int),
            }
        ),
        preserve_index=False,
    )

    def tokenize(batch: dict[str, list[str]]) -> dict[str, Any]:
        return tokenizer(batch["text"], truncation=True, max_length=MAX_LENGTH)

    return dataset.map(tokenize, batched=True).remove_columns(["text"])


def class_weights_for(labels: pd.Series, label2id: dict[str, int], max_weight: float = 20.0) -> torch.Tensor:
    counts = labels.value_counts().to_dict()
    total = float(sum(counts.values()))
    class_count = float(len(label2id))
    weights = []
    for label, _idx in sorted(label2id.items(), key=lambda item: item[1]):
        weight = total / max(class_count * float(counts.get(label, 1)), 1.0)
        weights.append(min(weight, max_weight))
    return torch.tensor(weights, dtype=torch.float)


class WeightedLossTrainer(Trainer):
    def __init__(self, *args: Any, class_weights: torch.Tensor | None = None, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: Any | None = None,
    ):
        del num_items_in_batch
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        weights = self.class_weights.to(logits.device) if self.class_weights is not None else None
        loss = nn.CrossEntropyLoss(weight=weights)(logits, labels)
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "macro_f1": f1_score(labels, predictions, average="macro", zero_division=0),
        "weighted_f1": f1_score(labels, predictions, average="weighted", zero_division=0),
    }


def training_arguments_kwargs(output_dir: Path) -> dict[str, Any]:
    import inspect

    params = inspect.signature(TrainingArguments.__init__).parameters
    kwargs: dict[str, Any] = {
        "output_dir": str(output_dir),
        "learning_rate": LEARNING_RATE,
        "per_device_train_batch_size": PER_DEVICE_TRAIN_BATCH_SIZE,
        "per_device_eval_batch_size": PER_DEVICE_EVAL_BATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "num_train_epochs": NUM_EPOCHS,
        "weight_decay": WEIGHT_DECAY,
        "logging_steps": 50,
        "save_total_limit": 2,
        "load_best_model_at_end": True,
        "metric_for_best_model": "macro_f1",
        "greater_is_better": True,
        "report_to": "none",
        "seed": SEED,
        "data_seed": SEED,
    }
    if "eval_strategy" in params:
        kwargs["eval_strategy"] = "epoch"
    else:
        kwargs["evaluation_strategy"] = "epoch"
    kwargs["save_strategy"] = "epoch"

    bf16_supported = bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    fp16_supported = bool(torch.cuda.is_available() and not bf16_supported)
    if "bf16" in params:
        kwargs["bf16"] = bf16_supported
    if "fp16" in params:
        kwargs["fp16"] = fp16_supported
    return kwargs

# %%!
def train_target(frame: pd.DataFrame, target: str) -> dict[str, Any]:
    print(f"Training target: {target}")
    target_dir = MODEL_DIR / f"{target}_model"
    checkpoint_dir = MODEL_DIR / f"{target}_checkpoints"
    target_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_df, valid_df, test_df = split_for_target(frame, target)
    label2id, id2label = make_label_maps(frame[target])

    train_dataset = build_dataset(train_df, target, label2id)
    valid_dataset = build_dataset(valid_df, target, label2id)
    test_dataset = build_dataset(test_df, target, label2id)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(label2id),
        id2label={int(k): v for k, v in id2label.items()},
        label2id=label2id,
    )

    import inspect

    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": TrainingArguments(**training_arguments_kwargs(checkpoint_dir)),
        "train_dataset": train_dataset,
        "eval_dataset": valid_dataset,
        "data_collator": DataCollatorWithPadding(tokenizer=tokenizer),
        "compute_metrics": compute_metrics,
        "class_weights": class_weights_for(train_df[target], label2id),
    }
    trainer_params = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = WeightedLossTrainer(**trainer_kwargs)

    trainer.train()
    validation_metrics = trainer.evaluate(valid_dataset, metric_key_prefix="validation")
    test_metrics = trainer.evaluate(test_dataset, metric_key_prefix="test")

    predictions = trainer.predict(test_dataset)
    predicted_ids = np.argmax(predictions.predictions, axis=-1)
    true_ids = predictions.label_ids
    label_names = [id2label[idx] for idx in range(len(id2label))]
    report = classification_report(
        true_ids,
        predicted_ids,
        labels=list(range(len(label_names))),
        target_names=label_names,
        zero_division=0,
        output_dict=True,
    )

    trainer.save_model(target_dir)
    tokenizer.save_pretrained(target_dir)

    artifact = {
        "target": target,
        "base_model": MODEL_NAME,
        "num_labels": len(label2id),
        "label2id": label2id,
        "id2label": {str(k): v for k, v in id2label.items()},
        "train_rows": int(len(train_df)),
        "validation_rows": int(len(valid_df)),
        "test_rows": int(len(test_df)),
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "classification_report": report,
    }
    with (target_dir / "training_artifact.json").open("w", encoding="utf-8") as handle:
        json.dump(artifact, handle, indent=2, ensure_ascii=False)

    del model
    del trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"Saved {target} model to {target_dir}")
    print(json.dumps(test_metrics, indent=2))
    return artifact


sector_artifact = train_target(df, "sector")
brand_artifact = train_target(df, "brand")

summary = {
    "sector": {
        "model_dir": str(MODEL_DIR / "sector_model"),
        "test_metrics": sector_artifact["test_metrics"],
    },
    "brand": {
        "model_dir": str(MODEL_DIR / "brand_model"),
        "test_metrics": brand_artifact["test_metrics"],
    },
}
with (MODEL_DIR / "training_summary.json").open("w", encoding="utf-8") as handle:
    json.dump(summary, handle, indent=2, ensure_ascii=False)

summary

# %%!
def load_classifier(model_path: Path):
    inference_tokenizer = AutoTokenizer.from_pretrained(model_path)
    inference_model = AutoModelForSequenceClassification.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inference_model.to(device)
    inference_model.eval()
    return inference_tokenizer, inference_model, device


def predict_one(model_path: Path, text: str, top_k: int = 5) -> list[dict[str, Any]]:
    inference_tokenizer, inference_model, device = load_classifier(model_path)
    encoded = inference_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
    ).to(device)
    with torch.no_grad():
        logits = inference_model(**encoded).logits[0]
        probabilities = torch.softmax(logits, dim=-1)
    k = min(top_k, probabilities.numel())
    values, indices = torch.topk(probabilities, k=k)
    return [
        {
            "label": inference_model.config.id2label[int(index.item())],
            "score": float(value.item()),
        }
        for value, index in zip(values.cpu(), indices.cpu())
    ]


sample = df.iloc[0]
sample_text = sample["text"]

{
    "url": sample["url"],
    "sector_predictions": predict_one(MODEL_DIR / "sector_model", sample_text),
    "brand_predictions": predict_one(MODEL_DIR / "brand_model", sample_text),
}
