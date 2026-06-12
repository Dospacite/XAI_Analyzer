# %% [markdown]
# # Gemma 4 E4B Phishing Classifier Fine-Tuning
#
# Colab workflow:
# 1. Mount Google Drive.
# 2. Load Stage B JSONL from Drive.
# 3. Build strict JSON-output SFT examples without exposing `label` or `source` in the model input.
# 4. Benchmark the raw model.
# 5. Fine-tune with Unsloth LoRA in bf16/16-bit mode.
# 6. Benchmark the fine-tuned adapter.
#
# Notes:
# - `unsloth/gemma-4-E4B-it-GGUF` is a GGUF/inference-format repo. Its Hugging Face model tree points to the trainable base
#   `google/gemma-4-E4B-it`, and Unsloth also publishes trainable safetensors as `unsloth/gemma-4-E4B-it`.
# - This script therefore fine-tunes `unsloth/gemma-4-E4B-it` in bf16/16-bit LoRA mode, then optionally exports after training.
# - Run in Google Colab A100 High-RAM. Start with small benchmark/test sizes, then scale once the pipeline is stable.

# %%
# @title Install dependencies
import os
import sys
import subprocess

INSTALL_DEPS = True  # @param {type:"boolean"}

if INSTALL_DEPS:
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--upgrade",
            "--force-reinstall",
            "--no-cache-dir",
            "unsloth",
            "unsloth_zoo",
        ]
    )
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--upgrade",
            "datasets",
            "transformers",
            "trl",
            "accelerate",
            "bitsandbytes",
            "tqdm",
        ]
    )

# %%
# @title Mount Google Drive and configure paths
from pathlib import Path

try:
    from google.colab import drive

    drive.mount("/content/drive")
except Exception as exc:
    print(f"Drive mount skipped or unavailable: {exc}")

DRIVE_PROJECT_DIR_STR = "/content/drive/MyDrive/XAI_Analyzer"  # @param {type:"string"}
DATA_JSONL_STR = "/content/drive/MyDrive/XAI_Analyzer/out_features.jsonl"  # @param {type:"string"}
RUN_DIR_STR = "/content/drive/MyDrive/XAI_Analyzer/gemma4_phishing_finetune"  # @param {type:"string"}

DRIVE_PROJECT_DIR = Path(DRIVE_PROJECT_DIR_STR)
DATA_JSONL = Path(DATA_JSONL_STR)
RUN_DIR = Path(RUN_DIR_STR)
REPORT_DIR = RUN_DIR / "reports"
MODEL_OUTPUT_DIR = RUN_DIR / "adapter"
MERGED_OUTPUT_DIR = RUN_DIR / "merged"
GGUF_OUTPUT_DIR = RUN_DIR / "gguf"

for path in (RUN_DIR, REPORT_DIR, MODEL_OUTPUT_DIR, MERGED_OUTPUT_DIR, GGUF_OUTPUT_DIR):
    path.mkdir(parents=True, exist_ok=True)

print("DATA_JSONL:", DATA_JSONL)
print("RUN_DIR:", RUN_DIR)

# %%
# @title Imports and global configuration
import gc
import json
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

import torch
from datasets import Dataset
from tqdm.auto import tqdm

from unsloth import FastModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only
from trl import SFTConfig, SFTTrainer


SEED = 3407
random.seed(SEED)
torch.manual_seed(SEED)

TRAINABLE_MODEL_CANDIDATES = [
    "unsloth/gemma-4-E4B-it",
    "google/gemma-4-E4B-it",
    "google/gemma-4-E4b-it",
]
GGUF_REFERENCE_REPO = "unsloth/gemma-4-E4B-it-GGUF"
MAX_SEQ_LENGTH = 8192  # @param {type:"integer"}
LOAD_IN_4BIT = True  # @param {type:"boolean"}

# Dataset sizing. Set MAX_RECORDS = 0 to read all JSONL lines.
MAX_RECORDS = 80000  # @param {type:"integer"}
MAX_TRAIN_EXAMPLES = 20000  # @param {type:"integer"}
MAX_EVAL_EXAMPLES_RAW = 50  # @param {type:"integer"}
MAX_EVAL_EXAMPLES_TUNED = 300  # @param {type:"integer"}
TEST_SIZE = 0.10  # @param {type:"number"}
VAL_SIZE = 0.05  # @param {type:"number"}

# SFT settings for A100 High-RAM. Increase max_steps or use num_train_epochs after smoke-testing.
PER_DEVICE_TRAIN_BATCH_SIZE = 2  # @param {type:"integer"}
GRADIENT_ACCUMULATION_STEPS = 4  # @param {type:"integer"}
LEARNING_RATE = 2e-4  # @param {type:"number"}
MAX_STEPS = 500  # @param {type:"integer"}
WARMUP_STEPS = 50  # @param {type:"integer"}
LOGGING_STEPS = 10  # @param {type:"integer"}
SAVE_STEPS = 250  # @param {type:"integer"}
DO_RAW_BENCHMARK = True  # @param {type:"boolean"}
DO_EVAL_DURING_TRAINING = False  # @param {type:"boolean"}
PACKING = True  # @param {type:"boolean"}

LORA_R = 16  # @param {type:"integer"}
LORA_ALPHA = 16  # @param {type:"integer"}

GEN_MAX_NEW_TOKENS = 256  # @param {type:"integer"}
EVAL_BATCH_SIZE = 4  # @param {type:"integer"}

EXPORT_MERGED_16BIT = False  # @param {type:"boolean"}
EXPORT_GGUF_BF16 = False  # @param {type:"boolean"}

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
print("Trainable model candidates:", TRAINABLE_MODEL_CANDIDATES)
print("GGUF reference:", GGUF_REFERENCE_REPO)

# %%
# @title Evidence policy
# These are targetable Stage B features. We intentionally exclude high-shortcut or noisy features like:
# - navigation.functional_internal_links
# - link.brand_text_domain_mismatch
# - form.hidden_inputs names
# - raw page-richness measurements as evidence
TARGETABLE_FEATURE_IDS = {
    "credential.password_input_present",
    "form.empty_or_blank_action",
    "form.external_action",
    "brand.title_domain_mismatch",
    "brand.domain_lookalike",
    "brand.favicon_domain_mismatch",
    "content.copyright_domain_mismatch",
    "content.coercive_urgency_near_form",
    "content.file_lure_terms",
    "contact.identity_domain_mismatch",
    "url.email_identifier",
    "url.homograph_or_unicode_hostname",
    "url.http_not_https",
    "redirect.cross_domain",
    "brand.title_domain_match_strong",
    "identity.provider_expected_domain",
    "support.contact_domain_match",
    "form.action_same_org_domain",
}

EXCLUDED_TARGET_FEATURE_IDS = {
    "navigation.functional_internal_links",
    "link.brand_text_domain_mismatch",
    "link.high_external_anchor_ratio",
    "form.hidden_inputs",
    "login.missing_recovery_or_help_flow",
    "page.low_semantic_content",
    "page.generic_login_without_brand_claim",
    "page.rendering_incomplete_or_script_dependent",
    "meta.robots_noindex_nofollow",
    "redirect.multi_hop",
    "url.long_url",
    "url.deep_subdomain",
}

SEVERITY_RANK = {"high": 3, "medium": 2, "low": 1}
MAX_EVIDENCE_ITEMS = 4


SYSTEM_PROMPT = (
    "You are a phishing website classifier. "
    "Return only valid JSON. Do not wrap it in Markdown. "
    "Use exactly this top-level structure: "
    '{"verdict":"phishing|benign|uncertain","confidence_level":"low|medium|high",'
    '"evidence":[{"id":"feature.id","direction":"suspicious|benign|neutral",'
    '"severity":"low|medium|high","value":{},"statement":"short human-readable reason"}]}. '
    "Allowed verdict values are phishing, benign, uncertain. "
    "Allowed confidence_level values are low, medium, high. "
    "Evidence items must use keys id, direction, severity, value, and statement. "
    "Use only observable artifacts from the provided website observation. "
    "Do not mention dataset source, training labels, collection names, or hidden metadata."
)

# %%
# @title JSON helpers, pruning, and target construction
def compact_text(value: Any, max_chars: int) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text[:max_chars].rstrip()


def redact_large_value(value: Any, max_chars: int = 240) -> Any:
    if isinstance(value, str):
        if value.startswith("data:"):
            return value[:48] + "<data_uri_truncated>"
        return compact_text(value, max_chars)
    if isinstance(value, list):
        return [redact_large_value(item, max_chars) for item in value[:20]]
    if isinstance(value, dict):
        return {key: redact_large_value(child, max_chars) for key, child in value.items()}
    return value


def prune_input(page_input: dict[str, Any]) -> dict[str, Any]:
    """Keep model-visible website observation compact and source/label free."""
    page_input = page_input or {}
    resources = page_input.get("resources") or {}
    pruned = {
        "url": compact_text(page_input.get("url"), 1000),
        "final_url": compact_text(page_input.get("final_url"), 1000),
        "redirects": redact_large_value(page_input.get("redirects", [])[:10]),
        "title": compact_text(page_input.get("title"), 300),
        "meta": redact_large_value((page_input.get("meta") or [])[:20], 240),
        "visible_text": compact_text(page_input.get("visible_text"), 2500),
        "forms": {
            "total_observed": (page_input.get("forms") or {}).get("total_observed", 0),
            "items": redact_large_value(((page_input.get("forms") or {}).get("items") or [])[:20], 240),
        },
        "anchors": {
            "total_observed": (page_input.get("anchors") or {}).get("total_observed", 0),
            "items": redact_large_value(((page_input.get("anchors") or {}).get("items") or [])[:30], 200),
        },
        "iframes": {
            "total_observed": (page_input.get("iframes") or {}).get("total_observed", 0),
            "items": redact_large_value(((page_input.get("iframes") or {}).get("items") or [])[:10], 200),
        },
        "resources": {
            "favicon_hrefs": redact_large_value((resources.get("favicon_hrefs") or [])[:10], 200),
            "script_src_sample": redact_large_value((resources.get("script_src_sample") or [])[:20], 200),
            "stylesheet_href_sample": redact_large_value((resources.get("stylesheet_href_sample") or [])[:20], 200),
            "image_src_sample": redact_large_value((resources.get("image_src_sample") or [])[:10], 120),
        },
    }
    return {key: value for key, value in pruned.items() if value not in ("", None)}


def sanitize_feature_for_target(feature: dict[str, Any]) -> dict[str, Any]:
    value = redact_large_value(feature.get("value") or {}, 180)
    if feature.get("id") == "form.hidden_inputs":
        value = {"count": value.get("count", 0) if isinstance(value, dict) else 0}
    return {
        "id": feature.get("id"),
        "direction": feature.get("direction"),
        "severity": feature.get("severity"),
        "value": value,
        "statement": compact_text(feature.get("statement"), 240),
    }


def select_target_evidence(label: str, features: list[dict[str, Any]]) -> list[dict[str, Any]]:
    verdict = "phishing" if label == "phishing" else "benign" if label == "benign" else "uncertain"
    wanted_direction = "suspicious" if verdict == "phishing" else "benign" if verdict == "benign" else "uncertain"
    candidates = []
    for feature in features or []:
        if not isinstance(feature, dict):
            continue
        feature_id = feature.get("id")
        if feature_id in EXCLUDED_TARGET_FEATURE_IDS:
            continue
        if feature_id not in TARGETABLE_FEATURE_IDS:
            continue
        if feature.get("direction") != wanted_direction:
            continue
        # Avoid turning weak neutral relationship artifacts into target evidence.
        if feature_id == "form.external_action":
            rel = (feature.get("value") or {}).get("relationship")
            if rel not in {"unrelated_third_party", "unknown"}:
                continue
        candidates.append(feature)

    candidates.sort(
        key=lambda f: (
            SEVERITY_RANK.get(str(f.get("severity")), 0),
            1 if not (f.get("supervision") or {}).get("primary_eligible") else 2,
            str(f.get("id")),
        ),
        reverse=True,
    )
    return [sanitize_feature_for_target(feature) for feature in candidates[:MAX_EVIDENCE_ITEMS]]


def confidence_from_evidence(label: str, evidence: list[dict[str, Any]]) -> str:
    if label not in {"phishing", "benign"}:
        return "low"
    if any(item.get("severity") == "high" for item in evidence) and len(evidence) >= 2:
        return "high"
    if evidence:
        return "medium"
    return "low"


def count_tokens_for_training_text(text: str) -> int:
    if hasattr(text_tokenizer, "encode"):
        return len(text_tokenizer.encode(text, add_special_tokens=False))
    encoded = text_tokenizer(text=text, add_special_tokens=False)
    input_ids = encoded["input_ids"]
    if input_ids and isinstance(input_ids[0], list):
        return len(input_ids[0])
    return len(input_ids)


def make_target_output(record: dict[str, Any]) -> dict[str, Any]:
    label = str(record.get("label") or "uncertain")
    verdict = "phishing" if label == "phishing" else "benign" if label == "benign" else "uncertain"
    evidence = select_target_evidence(label, record.get("features") or [])
    return {
        "verdict": verdict,
        "confidence_level": confidence_from_evidence(label, evidence),
        "evidence": evidence,
    }


def make_user_payload(record: dict[str, Any]) -> str:
    payload = {
        "task": "Classify the website as phishing, benign, or uncertain and return strict JSON.",
        "website_observation": prune_input(record.get("input") or {}),
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def extract_first_json_object(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
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
                        return json.loads(text[start : index + 1])
                    except json.JSONDecodeError:
                        return None
    return None


def normalize_verdict(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"phishing", "benign", "uncertain"}:
        return text
    if "phishing" in text or "phish" in text:
        return "phishing"
    if "benign" in text or "legitimate" in text or "safe" in text:
        return "benign"
    return "unparseable"


def stratified_split(records: list[dict[str, Any]], test_size: float, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    grouped = defaultdict(list)
    for record in records:
        grouped[record["label"]].append(record)

    left, right = [], []
    for items in grouped.values():
        items = items[:]
        rng.shuffle(items)
        if test_size <= 0:
            split_count = 0
        else:
            split_count = max(1, int(round(len(items) * test_size)))
            split_count = min(split_count, max(0, len(items) - 1))
        right.extend(items[:split_count])
        left.extend(items[split_count:])

    rng.shuffle(left)
    rng.shuffle(right)
    return left, right


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def metric_accuracy(y_true: list[str], y_pred: list[str]) -> float:
    return safe_div(sum(int(a == b) for a, b in zip(y_true, y_pred)), len(y_true))


def metric_confusion_matrix(y_true: list[str], y_pred: list[str], labels: list[str]) -> list[list[int]]:
    index = {label: offset for offset, label in enumerate(labels)}
    matrix = [[0 for _ in labels] for _ in labels]
    for expected, predicted in zip(y_true, y_pred):
        if expected in index and predicted in index:
            matrix[index[expected]][index[predicted]] += 1
    return matrix


def metric_precision_recall_f1(y_true: list[str], y_pred: list[str], label: str) -> tuple[float, float, float, int]:
    tp = sum(1 for expected, predicted in zip(y_true, y_pred) if expected == label and predicted == label)
    fp = sum(1 for expected, predicted in zip(y_true, y_pred) if expected != label and predicted == label)
    fn = sum(1 for expected, predicted in zip(y_true, y_pred) if expected == label and predicted != label)
    support = sum(1 for expected in y_true if expected == label)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    return precision, recall, f1, support


def metric_macro_f1(y_true: list[str], y_pred: list[str], labels: list[str]) -> float:
    return safe_div(sum(metric_precision_recall_f1(y_true, y_pred, label)[2] for label in labels), len(labels))


def make_classification_report(y_true: list[str], y_pred: list[str], labels: list[str]) -> str:
    lines = [f"{'label':<14}{'precision':>10}{'recall':>10}{'f1':>10}{'support':>10}"]
    for label in labels:
        precision, recall, f1, support = metric_precision_recall_f1(y_true, y_pred, label)
        lines.append(f"{label:<14}{precision:>10.4f}{recall:>10.4f}{f1:>10.4f}{support:>10}")
    lines.append(f"{'accuracy':<14}{metric_accuracy(y_true, y_pred):>30.4f}{len(y_true):>10}")
    return "\n".join(lines)

# %%
# @title Load and split dataset from Google Drive
def iter_jsonl(path: Path, max_records: int = 0):
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if max_records and index >= max_records:
                break
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


records = []
label_counts = Counter()
for record in tqdm(iter_jsonl(DATA_JSONL, MAX_RECORDS), desc="load_jsonl"):
    label = record.get("label")
    if label not in {"phishing", "benign"}:
        continue
    if not record.get("input"):
        continue
    label_counts[label] += 1
    records.append(record)

print("Loaded:", len(records), label_counts)
assert records, f"No records loaded from {DATA_JSONL}"

# Downsample to a balanced set to avoid label/source imbalance in training.
by_label = defaultdict(list)
for record in records:
    by_label[record["label"]].append(record)
min_label_count = min(len(by_label["phishing"]), len(by_label["benign"]))
balanced_records = []
for label in ("phishing", "benign"):
    random.shuffle(by_label[label])
    balanced_records.extend(by_label[label][:min_label_count])
random.shuffle(balanced_records)
print("Balanced:", len(balanced_records), Counter(r["label"] for r in balanced_records))

labels = [record["label"] for record in balanced_records]
train_records, test_records = stratified_split(balanced_records, TEST_SIZE, SEED)
val_fraction_of_train = VAL_SIZE / (1.0 - TEST_SIZE)
train_records, val_records = stratified_split(train_records, val_fraction_of_train, SEED + 1)

if MAX_TRAIN_EXAMPLES and len(train_records) > MAX_TRAIN_EXAMPLES:
    train_records = train_records[:MAX_TRAIN_EXAMPLES]

print("Split counts:")
print("train", len(train_records), Counter(r["label"] for r in train_records))
print("val", len(val_records), Counter(r["label"] for r in val_records))
print("test", len(test_records), Counter(r["label"] for r in test_records))

# %%
# @title Load raw Gemma 4 E4B model with Unsloth
last_load_error = None
for model_name in TRAINABLE_MODEL_CANDIDATES:
    try:
        print("Loading trainable model:", model_name)
        model, tokenizer = FastModel.from_pretrained(
            model_name=model_name,
            max_seq_length=MAX_SEQ_LENGTH,
            dtype=None,
            load_in_4bit=LOAD_IN_4BIT,
            full_finetuning=False,
        )
        TRAINABLE_MODEL_NAME = model_name
        break
    except Exception as exc:
        last_load_error = exc
        print(f"Could not load {model_name}: {type(exc).__name__}: {exc}")
else:
    raise RuntimeError("Could not load any trainable Gemma 4 model candidate.") from last_load_error

tokenizer = get_chat_template(tokenizer, chat_template="gemma-4-thinking")

text_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
if getattr(tokenizer, "pad_token", None) is None:
    tokenizer.pad_token = getattr(tokenizer, "eos_token", None) or getattr(text_tokenizer, "eos_token", None)
if getattr(text_tokenizer, "pad_token", None) is None:
    text_tokenizer.pad_token = getattr(text_tokenizer, "eos_token", None) or getattr(tokenizer, "eos_token", None)
if hasattr(tokenizer, "padding_side"):
    tokenizer.padding_side = "left"
if hasattr(text_tokenizer, "padding_side"):
    text_tokenizer.padding_side = "left"
if hasattr(tokenizer, "tokenizer") and hasattr(tokenizer.tokenizer, "padding_side"):
    tokenizer.tokenizer.padding_side = "left"
print("Processor padding_side:", getattr(tokenizer, "padding_side", None))
print("Text tokenizer padding_side:", getattr(text_tokenizer, "padding_side", None))

# %%
# @title Prompt formatting helpers
def make_messages(user_payload: str, assistant_payload: str | None = None) -> list[dict[str, Any]]:
    full_user_payload = f"{SYSTEM_PROMPT}\n\nWebsite observation:\n{user_payload}"
    messages = [{"role": "user", "content": [{"type": "text", "text": full_user_payload}]}]
    if assistant_payload is not None:
        messages.append({"role": "assistant", "content": [{"type": "text", "text": assistant_payload}]})
    return messages


def format_messages(user_payload: str, assistant_payload: str | None = None) -> str:
    return tokenizer.apply_chat_template(
        make_messages(user_payload, assistant_payload),
        tokenize=False,
        add_generation_prompt=assistant_payload is None,
    )


def record_to_training_text(record: dict[str, Any]) -> str:
    target = make_target_output(record)
    assistant_payload = json.dumps(target, ensure_ascii=False, separators=(",", ":"))
    return format_messages(make_user_payload(record), assistant_payload).removeprefix("<bos>")


def make_prompt(record: dict[str, Any]) -> str:
    return format_messages(make_user_payload(record), assistant_payload=None)


# Inspect one formatted example.
example_text = record_to_training_text(train_records[0])
print(example_text[:2500])
print("\nExample target:", json.dumps(make_target_output(train_records[0]), ensure_ascii=False, indent=2)[:2000])

# %%
# @title Build Hugging Face datasets
train_texts = [record_to_training_text(record) for record in tqdm(train_records, desc="format_train")]
val_texts = [record_to_training_text(record) for record in tqdm(val_records, desc="format_val")]
train_lengths = [count_tokens_for_training_text(text) for text in tqdm(train_texts[: min(1000, len(train_texts))], desc="length_probe")]
if train_lengths:
    print(
        {
            "length_probe_n": len(train_lengths),
            "max_seq_length": MAX_SEQ_LENGTH,
            "mean_tokens": round(sum(train_lengths) / len(train_lengths), 1),
            "p95_tokens": sorted(train_lengths)[int(0.95 * (len(train_lengths) - 1))],
            "max_tokens": max(train_lengths),
        }
    )
train_dataset = Dataset.from_dict({"text": train_texts})
val_dataset = Dataset.from_dict({"text": val_texts})

print(train_dataset)
print(val_dataset)

# %%
# @title Benchmark helper
@torch.inference_mode()
def generate_batch(records: list[dict[str, Any]]) -> list[tuple[str, dict[str, Any] | None]]:
    if hasattr(FastModel, "for_inference"):
        FastModel.for_inference(model)
    model.eval()
    prompts = [make_prompt(record) for record in records]
    inputs = text_tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
    ).to("cuda")
    prompt_width = inputs["input_ids"].shape[-1]
    outputs = model.generate(
        **inputs,
        max_new_tokens=GEN_MAX_NEW_TOKENS,
        do_sample=False,
        use_cache=True,
        pad_token_id=text_tokenizer.pad_token_id,
        eos_token_id=text_tokenizer.eos_token_id,
    )
    results = []
    for output in outputs:
        decoded = text_tokenizer.decode(output[prompt_width:], skip_special_tokens=True)
        results.append((decoded, extract_first_json_object(decoded)))
    return results


def generate_one(record: dict[str, Any]) -> tuple[str, dict[str, Any] | None]:
    return generate_batch([record])[0]


def benchmark(records_to_eval: list[dict[str, Any]], name: str, limit: int) -> dict[str, Any]:
    sample = records_to_eval[:limit] if limit else records_to_eval
    y_true, y_pred, parse_ok, rows = [], [], 0, []
    batch_size = max(1, int(EVAL_BATCH_SIZE))
    for start in tqdm(range(0, len(sample), batch_size), desc=f"benchmark_{name}"):
        batch = sample[start : start + batch_size]
        generated = generate_batch(batch)
        for record, (raw, parsed) in zip(batch, generated):
            predicted = normalize_verdict(parsed.get("verdict") if parsed else raw)
            expected = record["label"]
            y_true.append(expected)
            y_pred.append(predicted)
            parse_ok += int(parsed is not None)
            rows.append(
                {
                    "id": record.get("id"),
                    "label": expected,
                    "prediction": predicted,
                    "parsed": parsed is not None,
                    "raw_output": raw[:2000],
                    "target": make_target_output(record),
                }
            )
    labels_for_metrics = ["phishing", "benign"]
    report_labels = labels_for_metrics + ["uncertain", "unparseable"]
    metrics = {
        "name": name,
        "n": len(sample),
        "parse_rate": parse_ok / max(1, len(sample)),
        "accuracy": metric_accuracy(y_true, y_pred),
        "macro_f1": metric_macro_f1(y_true, y_pred, labels_for_metrics),
        "confusion_matrix": metric_confusion_matrix(y_true, y_pred, report_labels),
        "confusion_matrix_labels": report_labels,
        "classification_report": make_classification_report(y_true, y_pred, report_labels),
    }
    (REPORT_DIR / f"{name}_predictions.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    (REPORT_DIR / f"{name}_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in metrics.items() if k != "classification_report"}, indent=2))
    print(metrics["classification_report"])
    return metrics

# %%
# @title Benchmark raw model before fine-tuning
if DO_RAW_BENCHMARK and MAX_EVAL_EXAMPLES_RAW:
    raw_eval_records = test_records[:MAX_EVAL_EXAMPLES_RAW]
    raw_metrics = benchmark(raw_eval_records, "raw_model", MAX_EVAL_EXAMPLES_RAW)
else:
    raw_metrics = {"name": "raw_model", "n": 0, "skipped": True}

# %%
# @title Attach LoRA adapters
if hasattr(FastModel, "for_training"):
    FastModel.for_training(model)
else:
    model.train()

model = FastModel.get_peft_model(
    model,
    finetune_vision_layers=False,
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=0,
    bias="none",
    random_state=SEED,
)

# %%
# @title Fine-tune
def make_sft_config() -> SFTConfig:
    kwargs = {
        "max_seq_length": MAX_SEQ_LENGTH,
        "dataset_text_field": "text",
        "per_device_train_batch_size": PER_DEVICE_TRAIN_BATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "warmup_steps": WARMUP_STEPS,
        "max_steps": MAX_STEPS,
        "learning_rate": LEARNING_RATE,
        "logging_steps": LOGGING_STEPS,
        "save_steps": SAVE_STEPS,
        "eval_steps": SAVE_STEPS,
        "eval_strategy": "steps" if DO_EVAL_DURING_TRAINING else "no",
        "save_strategy": "steps",
        "output_dir": str(MODEL_OUTPUT_DIR),
        "optim": "adamw_8bit",
        "weight_decay": 0.001,
        "lr_scheduler_type": "linear",
        "seed": SEED,
        "dataset_num_proc": 2,
        "bf16": torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        "fp16": torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        "report_to": "none",
        "packing": PACKING,
    }
    for _ in range(4):
        try:
            return SFTConfig(**kwargs)
        except TypeError as exc:
            message = str(exc)
            if "eval_strategy" in message:
                kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")
                continue
            if "max_seq_length" in message:
                kwargs.pop("max_seq_length", None)
                continue
            if "packing" in message:
                kwargs.pop("packing", None)
                continue
            raise
    return SFTConfig(**kwargs)


sft_args = make_sft_config()

try:
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset if DO_EVAL_DURING_TRAINING else None,
        args=sft_args,
    )
except TypeError:
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset if DO_EVAL_DURING_TRAINING else None,
        args=sft_args,
    )

trainer = train_on_responses_only(
    trainer,
    instruction_part="<|turn>user\n",
    response_part="<|turn>model\n",
)

train_result = trainer.train()
trainer.save_model(str(MODEL_OUTPUT_DIR))
tokenizer.save_pretrained(str(MODEL_OUTPUT_DIR))
print(train_result)

# %%
# @title Benchmark fine-tuned model
gc.collect()
torch.cuda.empty_cache()

tuned_eval_records = test_records[:MAX_EVAL_EXAMPLES_TUNED]
tuned_metrics = benchmark(tuned_eval_records, "fine_tuned_model", MAX_EVAL_EXAMPLES_TUNED)

comparison = {"raw": raw_metrics, "fine_tuned": tuned_metrics}
(REPORT_DIR / "benchmark_comparison.json").write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({k: {m: v[m] for m in ("n", "parse_rate", "accuracy", "macro_f1")} for k, v in comparison.items()}, indent=2))

# %%
# @title Save merged model and optional BF16 GGUF
if EXPORT_MERGED_16BIT:
    model.save_pretrained_merged(str(MERGED_OUTPUT_DIR), tokenizer, save_method="merged_16bit")
    print("Saved merged 16-bit model to", MERGED_OUTPUT_DIR)

if EXPORT_GGUF_BF16:
    # This can take a while and may require llama.cpp build tooling under the hood.
    # If it fails, keep the LoRA adapter and merged 16-bit model; export separately.
    model.save_pretrained_gguf(str(GGUF_OUTPUT_DIR), tokenizer, quantization_method="bf16")
    print("Saved BF16 GGUF to", GGUF_OUTPUT_DIR)

# %%
# @title Quick manual inference on one held-out example
record = test_records[0]
raw, parsed = generate_one(record)
print("Expected label:", record["label"])
print("Parsed:", json.dumps(parsed, ensure_ascii=False, indent=2) if parsed else None)
print("Raw:", raw[:2000])
