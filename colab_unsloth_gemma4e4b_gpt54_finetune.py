#!/usr/bin/env python3
# %% [markdown]
# # Colab Unsloth Gemma 4 E4B Fine-Tuning on `gpt54_out_predictions.jsonl`
#
# Colab-oriented notebook-style script with `# %%` cells.
#
# This version is aligned to the current Unsloth Gemma 4 docs and the
# `unsloth/gemma-4-E4B-it-GGUF` Hugging Face model tree:
# - GGUF reference repo: `unsloth/gemma-4-E4B-it-GGUF`
# - Trainable base model from that tree: `google/gemma-4-E4B-it`
#
# Sources used for alignment:
# - Unsloth Gemma 4 run guide: https://unsloth.ai/docs/models/gemma-4
# - Unsloth Gemma 4 train guide: https://unsloth.ai/docs/models/gemma-4/train
# - HF GGUF repo / model tree: https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF
#
# Key alignment choices:
# - Uses standard Gemma 4 chat roles: `system`, `user`, `assistant`
# - Uses the Gemma 4 chat template
# - Uses docs-style text SFT on `google/gemma-4-E4B-it`
# - Uses bf16 / 16-bit LoRA by default (`load_in_16bit=True`, `load_in_4bit=False`)
# - Uses an H100 high-RAM oriented preset with a larger context / batch budget
# - Uses docs-style LoRA target modules and `use_gradient_checkpointing="unsloth"`
#
# Dataset behavior:
# - Input: `request.website_observation`
# - Target: `output_json`
# - Training keeps only rows where `label == output_json.verdict`
# - Reports metrics against both the original `label` and the teacher `output_json.verdict`

# %%
# @title Install dependencies
import subprocess
import sys

INSTALL_DEPS = True  # @param {type:"boolean"}

if INSTALL_DEPS:
    # TRL/Transformers can trip over a broken preinstalled wandb on Colab.
    # This pipeline does not use wandb, so remove it to avoid import-time failures.
    subprocess.call([sys.executable, "-m", "pip", "uninstall", "-y", "wandb"])
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--upgrade",
            "--force-reinstall",
            "--no-cache-dir",
            "--no-deps",
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
            "--force-reinstall",
            "--no-cache-dir",
            "--no-deps",
            "transformers",
            "timm",
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
            "trl",
            "accelerate",
            "bitsandbytes",
            "tqdm",
            "sentencepiece",
            "protobuf",
            "huggingface_hub",
        ]
    )

# %%
# @title Mount Google Drive and configure paths
import os
from pathlib import Path

os.environ["WANDB_DISABLED"] = "true"

try:
    from google.colab import drive

    drive.mount("/content/drive")
except Exception as exc:
    print(f"Drive mount skipped or unavailable: {exc}")

HF_TOKEN = ""  # @param {type:"string"}
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN

DRIVE_PROJECT_DIR_STR = "/content/drive/MyDrive/XAI_Analyzer"  # @param {type:"string"}
DATA_JSONL_STR = "/content/drive/MyDrive/XAI_Analyzer/gpt54_out_predictions.jsonl"  # @param {type:"string"}
RUN_DIR_STR = "/content/drive/MyDrive/XAI_Analyzer/gemma4e4b_gpt54_finetune"  # @param {type:"string"}

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
import random
import re
from collections import Counter, defaultdict
from typing import Any

os.environ.setdefault("UNSLOTH_STABLE_DOWNLOADS", "1")

import torch
from datasets import Dataset
from tqdm.auto import tqdm
from trl import SFTConfig, SFTTrainer

try:
    import unsloth as _unsloth
except Exception as exc:
    raise RuntimeError(
        "Could not import unsloth. Re-run the install cell, then restart the Colab runtime before importing training dependencies."
    ) from exc

FastLanguageModel = getattr(_unsloth, "FastLanguageModel", None) or getattr(_unsloth, "FastModel", None)
if FastLanguageModel is None:
    raise RuntimeError(
        "Installed unsloth does not expose FastLanguageModel or FastModel. "
        "Upgrade unsloth and unsloth_zoo to a Gemma 4 compatible version."
    )

from unsloth.chat_templates import get_chat_template, train_on_responses_only


SEED = 3407
random.seed(SEED)
torch.manual_seed(SEED)

GGUF_REFERENCE_REPO = "unsloth/gemma-4-E4B-it-GGUF"
TRAINABLE_MODEL_CANDIDATES = [
    "google/gemma-4-E4B-it",
    "unsloth/gemma-4-E4B-it",
]

MAX_SEQ_LENGTH = 8192  # @param {type:"integer"}
LOAD_IN_4BIT = False  # @param {type:"boolean"}
LOAD_IN_16BIT = True  # @param {type:"boolean"}

MAX_RECORDS = 0  # @param {type:"integer"}
MAX_TRAIN_EXAMPLES = 0  # @param {type:"integer"}
TEST_SIZE = 0.10  # @param {type:"number"}
VAL_SIZE = 0.05  # @param {type:"number"}
STRATIFY_BY = "label"  # @param ["label", "target_verdict"]
BALANCE_CLASSES = False  # @param {type:"boolean"}

ALIGN_TARGET_VERDICT_TO_LABEL = False  # @param {type:"boolean"}
USE_ONLY_CORRECT_PREDICTIONS = True  # @param {type:"boolean"}
MAX_EVIDENCE_ITEMS = 6  # @param {type:"integer"}
DROP_OVERSIZED_EXAMPLES = True  # @param {type:"boolean"}
RESPONSE_TRAINING_ONLY = True  # @param {type:"boolean"}
ENABLE_THINKING = False  # @param {type:"boolean"}

PER_DEVICE_TRAIN_BATCH_SIZE = 4  # @param {type:"integer"}
GRADIENT_ACCUMULATION_STEPS = 2  # @param {type:"integer"}
LEARNING_RATE = 2e-4  # @param {type:"number"}
MAX_STEPS = 300  # @param {type:"integer"}
WARMUP_STEPS = 10  # @param {type:"integer"}
LOGGING_STEPS = 1  # @param {type:"integer"}
SAVE_STEPS = 100  # @param {type:"integer"}
DO_EVAL_DURING_TRAINING = False  # @param {type:"boolean"}
PACKING = False  # @param {type:"boolean"}

LORA_R = 16  # @param {type:"integer"}
LORA_ALPHA = 16  # @param {type:"integer"}

DO_RAW_BENCHMARK = True  # @param {type:"boolean"}
MAX_EVAL_EXAMPLES_RAW = 50  # @param {type:"integer"}
MAX_EVAL_EXAMPLES_TUNED = 300  # @param {type:"integer"}
GEN_MAX_NEW_TOKENS = 512  # @param {type:"integer"}
EVAL_BATCH_SIZE = 8  # @param {type:"integer"}
PROMPT_TOKEN_BUDGET = 6144  # @param {type:"integer"}

EXPORT_MERGED_16BIT = False  # @param {type:"boolean"}
EXPORT_GGUF_BF16 = False  # @param {type:"boolean"}

GEMMA_CHAT_TEMPLATE = "gemma-4-thinking"

BASE_SYSTEM_PROMPT = (
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

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
print("GGUF reference repo:", GGUF_REFERENCE_REPO)
print("Trainable model candidates:", TRAINABLE_MODEL_CANDIDATES)

# %%
# @title Configuration validation helpers
def validate_runtime_configuration() -> None:
    if LOAD_IN_4BIT and LOAD_IN_16BIT:
        raise ValueError(
            "LOAD_IN_4BIT and LOAD_IN_16BIT cannot both be True. "
            "Unsloth allows only one training load mode at a time."
        )
    if not LOAD_IN_4BIT and not LOAD_IN_16BIT:
        print("Warning: both LOAD_IN_4BIT and LOAD_IN_16BIT are False; this may require much more VRAM.")
    if MAX_SEQ_LENGTH <= 0:
        raise ValueError(f"MAX_SEQ_LENGTH must be positive, got {MAX_SEQ_LENGTH}.")
    if PROMPT_TOKEN_BUDGET <= 0:
        raise ValueError(f"PROMPT_TOKEN_BUDGET must be positive, got {PROMPT_TOKEN_BUDGET}.")
    if TEST_SIZE < 0 or TEST_SIZE >= 1:
        raise ValueError(f"TEST_SIZE must be in [0, 1), got {TEST_SIZE}.")
    if VAL_SIZE < 0 or VAL_SIZE >= 1:
        raise ValueError(f"VAL_SIZE must be in [0, 1), got {VAL_SIZE}.")
    if TEST_SIZE + VAL_SIZE >= 1:
        raise ValueError(
            f"TEST_SIZE + VAL_SIZE must be < 1, got {TEST_SIZE + VAL_SIZE:.4f}. "
            "Otherwise there is no training split left."
        )
    if EVAL_BATCH_SIZE <= 0:
        raise ValueError(f"EVAL_BATCH_SIZE must be positive, got {EVAL_BATCH_SIZE}.")
    if ENABLE_THINKING and RESPONSE_TRAINING_ONLY:
        print(
            "Warning: ENABLE_THINKING=True while RESPONSE_TRAINING_ONLY=True. "
            "The Gemma 4 guide recommends keeping thinking-mode formatting consistent; "
            "this notebook trains on final visible answers only by default."
        )


validate_runtime_configuration()

# %%
# @title JSON helpers
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


def collection_items(value: Any) -> list[Any]:
    if isinstance(value, dict):
        items = value.get("items") or []
        return items if isinstance(items, list) else []
    if isinstance(value, list):
        return value
    return []


def collection_total(value: Any) -> int:
    if isinstance(value, dict):
        total = value.get("total_observed")
        if isinstance(total, int):
            return total
        return len(collection_items(value))
    if isinstance(value, list):
        return len(value)
    return 0


def prune_input(page_input: dict[str, Any]) -> dict[str, Any]:
    page_input = page_input or {}
    resources = page_input.get("resources") or {}
    forms = page_input.get("forms")
    anchors = page_input.get("anchors")
    iframes = page_input.get("iframes")
    pruned = {
        "url": compact_text(page_input.get("url"), 1000),
        "final_url": compact_text(page_input.get("final_url"), 1000),
        "redirects": redact_large_value((page_input.get("redirects") or [])[:10]),
        "title": compact_text(page_input.get("title"), 300),
        "meta": redact_large_value((page_input.get("meta") or [])[:20], 240),
        "visible_text": compact_text(page_input.get("visible_text"), 1200),
        "forms": {
            "total_observed": collection_total(forms),
            "items": redact_large_value(collection_items(forms)[:10], 200),
        },
        "anchors": {
            "total_observed": collection_total(anchors),
            "items": redact_large_value(collection_items(anchors)[:12], 160),
        },
        "iframes": {
            "total_observed": collection_total(iframes),
            "items": redact_large_value(collection_items(iframes)[:6], 160),
        },
        "resources": {
            "favicon_hrefs": redact_large_value((resources.get("favicon_hrefs") or [])[:4], 160),
            "script_src_sample": redact_large_value((resources.get("script_src_sample") or [])[:8], 160),
            "stylesheet_href_sample": redact_large_value((resources.get("stylesheet_href_sample") or [])[:6], 160),
            "image_src_sample": redact_large_value((resources.get("image_src_sample") or [])[:4], 100),
        },
    }
    return {key: value for key, value in pruned.items() if value not in ("", None)}


def sanitize_evidence_item(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": compact_text(item.get("id"), 120),
        "direction": compact_text(item.get("direction"), 32),
        "severity": compact_text(item.get("severity"), 16),
        "value": redact_large_value(item.get("value"), 180),
        "statement": compact_text(item.get("statement"), 240),
    }


def normalize_verdict(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"phishing", "benign", "uncertain"}:
        return text
    if "phish" in text:
        return "phishing"
    if "benign" in text or "legitimate" in text or "safe" in text:
        return "benign"
    if "uncertain" in text or "unknown" in text:
        return "uncertain"
    return "unparseable"


def normalize_confidence(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"low", "medium", "high"}:
        return text
    match = re.search(r"\bconfidence(?:_level)?\b[^a-z]{0,20}\b(low|medium|high)\b", text)
    if match:
        return match.group(1)
    for level in ("high", "medium", "low"):
        if re.search(rf"\b{level}\s+confidence\b", text):
            return level
    return "low"


def normalize_direction(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"suspicious", "benign", "neutral"}:
        return text
    if any(token in text for token in ("suspicious", "phish", "malicious", "credential")):
        return "suspicious"
    if any(token in text for token in ("benign", "legitimate", "safe", "trusted")):
        return "benign"
    return "neutral"


def normalize_prediction_payload(payload: Any) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    verdict = normalize_verdict(payload.get("verdict"))
    if verdict == "unparseable":
        return None
    evidence = []
    for offset, item in enumerate(payload.get("evidence") or []):
        if not isinstance(item, dict):
            continue
        statement = compact_text(item.get("statement"), 240)
        if not statement:
            continue
        evidence.append(
            {
                "id": compact_text(item.get("id"), 120) or f"model.evidence.{offset + 1}",
                "direction": normalize_direction(item.get("direction")),
                "severity": normalize_confidence(item.get("severity")),
                "value": redact_large_value(item.get("value"), 180),
                "statement": statement,
            }
        )
        if len(evidence) >= MAX_EVIDENCE_ITEMS:
            break
    return {
        "verdict": verdict,
        "confidence_level": normalize_confidence(payload.get("confidence_level")),
        "evidence": evidence,
    }


def infer_verdict_from_text(text: str) -> str:
    lowered = str(text or "").strip().lower()
    if not lowered:
        return "unparseable"
    patterns = [
        r"\bverdict\b[^a-z]{0,20}\b(phishing|benign|uncertain)\b",
        r"\b(?:classification|classify|classified|conclusion|final answer|final classification|overall assessment)\b[^.\n]{0,160}\b(phishing|benign|uncertain)\b",
        r"\b(?:this|it|page|site|website)\s+(?:is|looks)\s+(phishing|benign|uncertain)\b",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, lowered, flags=re.IGNORECASE)
        if matches:
            return str(matches[-1]).lower()
    tail_matches = re.findall(r"\b(phishing|benign|uncertain)\b", lowered[-800:], flags=re.IGNORECASE)
    if tail_matches:
        return str(tail_matches[-1]).lower()
    return "unparseable"


def repair_prediction_payload(text: str) -> dict[str, Any] | None:
    verdict = infer_verdict_from_text(text)
    if verdict == "unparseable":
        return None
    return {
        "verdict": verdict,
        "confidence_level": normalize_confidence(text),
        "evidence": [],
    }


def make_system_prompt() -> str:
    if ENABLE_THINKING:
        return "<|think|>\n" + BASE_SYSTEM_PROMPT
    return BASE_SYSTEM_PROMPT


def make_target_output(record: dict[str, Any]) -> dict[str, Any]:
    output_json = record.get("output_json") or {}
    verdict = normalize_verdict(output_json.get("verdict"))
    if ALIGN_TARGET_VERDICT_TO_LABEL and record.get("label") in {"phishing", "benign", "uncertain"}:
        verdict = str(record["label"])
    evidence = [
        sanitize_evidence_item(item)
        for item in (output_json.get("evidence") or [])
        if isinstance(item, dict)
    ][:MAX_EVIDENCE_ITEMS]
    return {
        "verdict": verdict if verdict != "unparseable" else "uncertain",
        "confidence_level": normalize_confidence(output_json.get("confidence_level")),
        "evidence": evidence,
    }


def make_user_payload(record: dict[str, Any]) -> str:
    request = record.get("request") or {}
    payload = {
        "task": request.get("task") or "Classify the website as phishing, benign, or uncertain and return strict JSON.",
        "website_observation": prune_input(request.get("website_observation") or {}),
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


def strip_gemma4_reasoning_channels(text: str) -> str:
    cleaned = str(text or "")
    pattern = re.compile(r"^\s*<\|channel\|>thought\s*.*?<channel\|>\s*", flags=re.DOTALL)
    while True:
        updated = pattern.sub("", cleaned, count=1)
        if updated == cleaned:
            return cleaned.strip()
        cleaned = updated


def parse_prediction_output(text: str) -> tuple[dict[str, Any] | None, str]:
    visible_text = strip_gemma4_reasoning_channels(text)
    strict = normalize_prediction_payload(extract_first_json_object(visible_text))
    if strict is not None:
        return strict, "strict_json"
    repaired = repair_prediction_payload(visible_text)
    if repaired is not None:
        return repaired, "heuristic_repair"
    return None, "unparseable"


def count_tokens_for_training_text(text: str) -> int:
    if hasattr(text_tokenizer, "encode"):
        return len(text_tokenizer.encode(text, add_special_tokens=False))
    encoded = text_tokenizer(text=text, add_special_tokens=False)
    input_ids = encoded["input_ids"]
    if input_ids and isinstance(input_ids[0], list):
        return len(input_ids[0])
    return len(input_ids)

# %%
# @title Metrics helpers
def stratified_split(records: list[dict[str, Any]], test_size: float, seed: int, key_name: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    grouped = defaultdict(list)
    for record in records:
        grouped[str(record[key_name])].append(record)

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
# @title Load and split dataset
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
teacher_mismatch_count = 0
filtered_incorrect_predictions = 0
if not DATA_JSONL.is_file():
    raise FileNotFoundError(f"DATA_JSONL does not exist: {DATA_JSONL}")

for record in tqdm(iter_jsonl(DATA_JSONL, MAX_RECORDS), desc="load_jsonl"):
    label = normalize_verdict(record.get("label"))
    request = record.get("request") or {}
    website_observation = request.get("website_observation") or {}
    target = make_target_output(record)
    target_verdict = target["verdict"]
    if label not in {"phishing", "benign"}:
        continue
    if target_verdict not in {"phishing", "benign", "uncertain"}:
        continue
    if not website_observation:
        continue
    if USE_ONLY_CORRECT_PREDICTIONS and label != target_verdict:
        filtered_incorrect_predictions += 1
        continue
    prepared = {
        **record,
        "label": label,
        "target_output": target,
        "target_verdict": target_verdict,
    }
    teacher_mismatch_count += int(label != target_verdict)
    records.append(prepared)

assert records, f"No usable rows found in {DATA_JSONL}"
print("Loaded rows:", len(records))
print("Label counts:", Counter(record["label"] for record in records))
print("Target verdict counts:", Counter(record["target_verdict"] for record in records))
print("Label/target verdict mismatches:", teacher_mismatch_count)
print("Filtered incorrect predictions:", filtered_incorrect_predictions)

split_key = "target_verdict" if STRATIFY_BY == "target_verdict" else "label"

if BALANCE_CLASSES:
    grouped = defaultdict(list)
    for record in records:
        grouped[str(record[split_key])].append(record)
    class_counts = {key: len(grouped.get(key, [])) for key in ("phishing", "benign")}
    if not all(class_counts.values()):
        raise RuntimeError(
            "BALANCE_CLASSES=True requires both phishing and benign examples after filtering. "
            f"Observed counts: {class_counts}"
        )
    min_count = min(class_counts.values())
    selected_records = []
    for key, items in grouped.items():
        if key in {"phishing", "benign"}:
            random.shuffle(items)
            selected_records.extend(items[:min_count])
    random.shuffle(selected_records)
else:
    selected_records = records[:]

train_records, test_records = stratified_split(selected_records, TEST_SIZE, SEED, split_key)
val_fraction_of_train = VAL_SIZE / (1.0 - TEST_SIZE)
train_records, val_records = stratified_split(train_records, val_fraction_of_train, SEED + 1, split_key)

if MAX_TRAIN_EXAMPLES and len(train_records) > MAX_TRAIN_EXAMPLES:
    train_records = train_records[:MAX_TRAIN_EXAMPLES]

if not train_records:
    raise RuntimeError(
        "No training rows remain after filtering and splitting. "
        "Reduce TEST_SIZE / VAL_SIZE, disable aggressive filtering, or inspect the dataset."
    )

print("train", len(train_records), Counter(record[split_key] for record in train_records))
print("val", len(val_records), Counter(record[split_key] for record in val_records))
print("test", len(test_records), Counter(record[split_key] for record in test_records))

(RUN_DIR / "dataset_summary.json").write_text(
    json.dumps(
        {
            "rows": len(records),
            "label_counts": Counter(record["label"] for record in records),
            "target_verdict_counts": Counter(record["target_verdict"] for record in records),
            "teacher_mismatch_count": teacher_mismatch_count,
            "filtered_incorrect_predictions": filtered_incorrect_predictions,
            "train_rows": len(train_records),
            "val_rows": len(val_records),
            "test_rows": len(test_records),
        },
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)

# %%
# @title Load docs-aligned Gemma 4 E4B base model
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required for this Colab Unsloth Gemma 4 pipeline.")

last_load_error = None
for model_name in TRAINABLE_MODEL_CANDIDATES:
    try:
        print("Loading model:", model_name)
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=MAX_SEQ_LENGTH,
            load_in_4bit=LOAD_IN_4BIT,
            load_in_16bit=LOAD_IN_16BIT,
            full_finetuning=False,
        )
        TRAINABLE_MODEL_NAME = model_name
        break
    except Exception as exc:
        last_load_error = exc
        print(f"Could not load {model_name}: {type(exc).__name__}: {exc}")
else:
    raise RuntimeError(
        "Could not load any Gemma 4 E4B trainable model. "
        "The GGUF repo is a reference/inference repo; training uses its base model tree."
    ) from last_load_error

tokenizer = get_chat_template(tokenizer, chat_template=GEMMA_CHAT_TEMPLATE)


def configure_tokenizers(padding_side: str) -> None:
    if padding_side not in {"left", "right"}:
        raise ValueError(f"Unsupported padding_side: {padding_side}")
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = getattr(tokenizer, "eos_token", None) or getattr(text_tokenizer, "eos_token", None)
    if getattr(text_tokenizer, "pad_token", None) is None:
        text_tokenizer.pad_token = getattr(text_tokenizer, "eos_token", None) or getattr(tokenizer, "eos_token", None)
    if hasattr(tokenizer, "padding_side"):
        tokenizer.padding_side = padding_side
    if hasattr(text_tokenizer, "padding_side"):
        text_tokenizer.padding_side = padding_side
    if hasattr(tokenizer, "tokenizer") and hasattr(tokenizer.tokenizer, "padding_side"):
        tokenizer.tokenizer.padding_side = padding_side
    if hasattr(tokenizer, "tokenizer") and getattr(tokenizer.tokenizer, "pad_token", None) is None:
        tokenizer.tokenizer.pad_token = getattr(tokenizer.tokenizer, "eos_token", None) or getattr(tokenizer, "eos_token", None)


text_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
configure_tokenizers("right")

print("Loaded trainable model:", TRAINABLE_MODEL_NAME)
print("Backed by GGUF reference tree:", GGUF_REFERENCE_REPO)

# %%
# @title Format Gemma 4 chat data
def make_messages(user_payload: str, assistant_payload: str | None = None) -> list[dict[str, str]]:
    messages = [
        {"role": "system", "content": make_system_prompt()},
        {"role": "user", "content": user_payload},
    ]
    if assistant_payload is not None:
        messages.append({"role": "assistant", "content": assistant_payload})
    return messages


def format_messages(user_payload: str, assistant_payload: str | None = None) -> str:
    return tokenizer.apply_chat_template(
        make_messages(user_payload, assistant_payload),
        tokenize=False,
        add_generation_prompt=assistant_payload is None,
    )


def record_to_training_text(record: dict[str, Any]) -> str:
    assistant_payload = json.dumps(record["target_output"], ensure_ascii=False, separators=(",", ":"))
    return format_messages(make_user_payload(record), assistant_payload)


def make_prompt(record: dict[str, Any]) -> str:
    return format_messages(make_user_payload(record), assistant_payload=None)


example_text = record_to_training_text(train_records[0])
print(example_text[:2500])
print(json.dumps(train_records[0]["target_output"], ensure_ascii=False, indent=2)[:1500])


def infer_response_only_markers(formatted_text: str) -> tuple[str, str]:
    matches = list(
        re.finditer(
            r"(<\|turn\|>|<\|turn>|<start_of_turn>)(system|user|assistant|model)\n",
            formatted_text,
        )
    )
    if len(matches) < 2:
        raise RuntimeError(
            "Could not infer Gemma 4 turn markers from the formatted chat text. "
            "Inspect the active chat template before using response-only training."
        )
    response_index = next((index for index in range(len(matches) - 1, -1, -1) if matches[index].group(2) in {"assistant", "model"}), None)
    if response_index is None or response_index == 0:
        raise RuntimeError(
            "Could not find a final assistant/model turn marker in the formatted chat text."
        )
    instruction_match = matches[response_index - 1]
    response_match = matches[response_index]
    if instruction_match.group(2) != "user":
        raise RuntimeError(
            "Gemma 4 response-only training expects the assistant reply to follow a user turn. "
            f"Observed `{instruction_match.group(2)}` before the assistant turn instead."
        )
    return instruction_match.group(0), response_match.group(0)


RESPONSE_ONLY_INSTRUCTION_PART, RESPONSE_ONLY_RESPONSE_PART = infer_response_only_markers(example_text)
print("Response-only instruction marker:", repr(RESPONSE_ONLY_INSTRUCTION_PART))
print("Response-only response marker:", repr(RESPONSE_ONLY_RESPONSE_PART))

# %%
# @title Build Hugging Face datasets
train_prompts = [make_prompt(record) for record in tqdm(train_records, desc="prompt_train")]
val_prompts = [make_prompt(record) for record in tqdm(val_records, desc="prompt_val")]
test_prompts = [make_prompt(record) for record in tqdm(test_records, desc="prompt_test")]

train_texts = [record_to_training_text(record) for record in tqdm(train_records, desc="format_train")]
val_texts = [record_to_training_text(record) for record in tqdm(val_records, desc="format_val")]

train_prompt_lengths = [count_tokens_for_training_text(text) for text in tqdm(train_prompts, desc="prompt_tokens_train")]
val_prompt_lengths = [count_tokens_for_training_text(text) for text in tqdm(val_prompts, desc="prompt_tokens_val")]
test_prompt_lengths = [count_tokens_for_training_text(text) for text in tqdm(test_prompts, desc="prompt_tokens_test")]

train_token_lengths = [count_tokens_for_training_text(text) for text in tqdm(train_texts, desc="tokens_train")]
val_token_lengths = [count_tokens_for_training_text(text) for text in tqdm(val_texts, desc="tokens_val")]

if DROP_OVERSIZED_EXAMPLES:
    filtered_train = [
        (record, text, prompt_tokens, total_tokens)
        for record, text, prompt_tokens, total_tokens in zip(train_records, train_texts, train_prompt_lengths, train_token_lengths)
        if total_tokens <= MAX_SEQ_LENGTH and prompt_tokens <= PROMPT_TOKEN_BUDGET
    ]
    filtered_val = [
        (record, text, prompt_tokens, total_tokens)
        for record, text, prompt_tokens, total_tokens in zip(val_records, val_texts, val_prompt_lengths, val_token_lengths)
        if total_tokens <= MAX_SEQ_LENGTH and prompt_tokens <= PROMPT_TOKEN_BUDGET
    ]
    filtered_test = [
        (record, prompt_tokens)
        for record, prompt_tokens in zip(test_records, test_prompt_lengths)
        if prompt_tokens <= PROMPT_TOKEN_BUDGET
    ]
    dropped_train = len(train_texts) - len(filtered_train)
    dropped_val = len(val_texts) - len(filtered_val)
    dropped_test = len(test_records) - len(filtered_test)
    train_records = [record for record, _, _, _ in filtered_train]
    val_records = [record for record, _, _, _ in filtered_val]
    test_records = [record for record, _ in filtered_test]
    train_texts = [text for _, text, _, _ in filtered_train]
    val_texts = [text for _, text, _, _ in filtered_val]
    train_prompt_lengths = [prompt_tokens for _, _, prompt_tokens, _ in filtered_train]
    val_prompt_lengths = [prompt_tokens for _, _, prompt_tokens, _ in filtered_val]
    train_token_lengths = [total_tokens for _, _, _, total_tokens in filtered_train]
    val_token_lengths = [total_tokens for _, _, _, total_tokens in filtered_val]
    print("Dropped oversized train rows:", dropped_train)
    print("Dropped oversized val rows:", dropped_val)
    print("Dropped oversized test rows:", dropped_test)
    if not train_texts:
        raise RuntimeError("All training rows exceeded MAX_SEQ_LENGTH after formatting. Increase MAX_SEQ_LENGTH or prune inputs harder.")

if train_token_lengths:
    print(
        {
            "length_probe_n": len(train_token_lengths),
            "max_seq_length": MAX_SEQ_LENGTH,
            "prompt_token_budget": PROMPT_TOKEN_BUDGET,
            "mean_prompt_tokens": round(sum(train_prompt_lengths) / len(train_prompt_lengths), 1),
            "p95_prompt_tokens": sorted(train_prompt_lengths)[int(0.95 * (len(train_prompt_lengths) - 1))],
            "max_prompt_tokens": max(train_prompt_lengths),
            "mean_tokens": round(sum(train_token_lengths) / len(train_token_lengths), 1),
            "p95_tokens": sorted(train_token_lengths)[int(0.95 * (len(train_token_lengths) - 1))],
            "max_tokens": max(train_token_lengths),
        }
    )

train_dataset = Dataset.from_dict({"text": train_texts})
val_dataset = Dataset.from_dict({"text": val_texts})
print(train_dataset)
print(val_dataset)

# %%
# @title Benchmark helper
@torch.inference_mode()
def generate_batch(records: list[dict[str, Any]]) -> list[tuple[str, dict[str, Any] | None, str]]:
    if not records:
        return []
    if hasattr(FastLanguageModel, "for_inference"):
        FastLanguageModel.for_inference(model)
    model.eval()
    configure_tokenizers("left")
    try:
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
            temperature=1.0,
            top_p=0.95,
            top_k=64,
            use_cache=True,
            pad_token_id=text_tokenizer.pad_token_id,
            eos_token_id=text_tokenizer.eos_token_id,
        )
        results = []
        for output in outputs:
            decoded = text_tokenizer.decode(output[prompt_width:], skip_special_tokens=True)
            parsed, parse_mode = parse_prediction_output(decoded)
            results.append((decoded, parsed, parse_mode))
        return results
    finally:
        configure_tokenizers("right")


def generate_one(record: dict[str, Any]) -> tuple[str, dict[str, Any] | None, str]:
    return generate_batch([record])[0]


def benchmark(records_to_eval: list[dict[str, Any]], name: str, limit: int) -> dict[str, Any]:
    sample = records_to_eval[:limit] if limit else records_to_eval
    y_pred, y_true_label, y_true_target, rows = [], [], [], []
    parse_ok = 0
    structured_ok = 0
    batch_size = max(1, int(EVAL_BATCH_SIZE))
    for start in tqdm(range(0, len(sample), batch_size), desc=f"benchmark_{name}"):
        batch = sample[start : start + batch_size]
        generated = generate_batch(batch)
        for record, (raw, parsed, parse_mode) in zip(batch, generated):
            predicted = normalize_verdict(parsed.get("verdict") if parsed else raw)
            expected_label = record["label"]
            expected_target = record["target_verdict"]
            y_pred.append(predicted)
            y_true_label.append(expected_label)
            y_true_target.append(expected_target)
            parse_ok += int(parse_mode == "strict_json")
            structured_ok += int(parsed is not None)
            rows.append(
                {
                    "id": record.get("id"),
                    "label": expected_label,
                    "target_verdict": expected_target,
                    "prediction": predicted,
                    "parsed": parsed is not None,
                    "parse_mode": parse_mode,
                    "raw_output": raw[:2000],
                    "target_output": record["target_output"],
                }
            )

    primary_labels = ["phishing", "benign"]
    report_labels = primary_labels + ["uncertain", "unparseable"]
    metrics = {
        "name": name,
        "n": len(sample),
        "parse_rate": safe_div(parse_ok, len(sample)),
        "structured_parse_rate": safe_div(structured_ok, len(sample)),
        "accuracy_vs_label": metric_accuracy(y_true_label, y_pred),
        "macro_f1_vs_label": metric_macro_f1(y_true_label, y_pred, primary_labels),
        "accuracy_vs_target": metric_accuracy(y_true_target, y_pred),
        "macro_f1_vs_target": metric_macro_f1(y_true_target, y_pred, primary_labels),
        "confusion_vs_label": metric_confusion_matrix(y_true_label, y_pred, report_labels),
        "confusion_vs_target": metric_confusion_matrix(y_true_target, y_pred, report_labels),
        "confusion_labels": report_labels,
        "classification_report_vs_label": make_classification_report(y_true_label, y_pred, report_labels),
        "classification_report_vs_target": make_classification_report(y_true_target, y_pred, report_labels),
    }
    (REPORT_DIR / f"{name}_predictions.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    (REPORT_DIR / f"{name}_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "name": metrics["name"],
                "n": metrics["n"],
                "parse_rate": metrics["parse_rate"],
                "structured_parse_rate": metrics["structured_parse_rate"],
                "accuracy_vs_label": metrics["accuracy_vs_label"],
                "macro_f1_vs_label": metrics["macro_f1_vs_label"],
                "accuracy_vs_target": metrics["accuracy_vs_target"],
                "macro_f1_vs_target": metrics["macro_f1_vs_target"],
            },
            indent=2,
        )
    )
    return metrics

# %%
# @title Benchmark raw model before fine-tuning
if DO_RAW_BENCHMARK and MAX_EVAL_EXAMPLES_RAW:
    raw_metrics = benchmark(test_records, "raw_model", MAX_EVAL_EXAMPLES_RAW)
else:
    raw_metrics = {"name": "raw_model", "n": 0, "skipped": True}

# %%
# @title Attach LoRA adapters using docs-aligned settings
if hasattr(FastLanguageModel, "for_training"):
    FastLanguageModel.for_training(model)
else:
    model.train()
configure_tokenizers("right")

model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_R,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_alpha=LORA_ALPHA,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=SEED,
    max_seq_length=MAX_SEQ_LENGTH,
)

# %%
# @title Fine-tune
has_eval_data = DO_EVAL_DURING_TRAINING and len(val_dataset) > 0
if DO_EVAL_DURING_TRAINING and not has_eval_data:
    print("Evaluation during training was requested, but no validation rows remained after filtering. Disabling eval during training.")


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
        "eval_strategy": "steps" if has_eval_data else "no",
        "save_strategy": "steps",
        "output_dir": str(MODEL_OUTPUT_DIR),
        "optim": "adamw_8bit",
        "seed": SEED,
        "dataset_num_proc": 1,
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
eval_dataset = val_dataset if has_eval_data else None

try:
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=sft_args,
    )
except TypeError:
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=sft_args,
    )

if RESPONSE_TRAINING_ONLY:
    trainer = train_on_responses_only(
        trainer,
        instruction_part=RESPONSE_ONLY_INSTRUCTION_PART,
        response_part=RESPONSE_ONLY_RESPONSE_PART,
    )

(RUN_DIR / "training_config.json").write_text(
    json.dumps(
        {
            "gguf_reference_repo": GGUF_REFERENCE_REPO,
            "loaded_model_name": TRAINABLE_MODEL_NAME,
            "max_seq_length": MAX_SEQ_LENGTH,
            "load_in_4bit": LOAD_IN_4BIT,
            "load_in_16bit": LOAD_IN_16BIT,
            "max_records": MAX_RECORDS,
            "max_train_examples": MAX_TRAIN_EXAMPLES,
            "test_size": TEST_SIZE,
            "val_size": VAL_SIZE,
            "stratify_by": STRATIFY_BY,
            "balance_classes": BALANCE_CLASSES,
            "align_target_verdict_to_label": ALIGN_TARGET_VERDICT_TO_LABEL,
            "use_only_correct_predictions": USE_ONLY_CORRECT_PREDICTIONS,
            "drop_oversized_examples": DROP_OVERSIZED_EXAMPLES,
            "response_training_only": RESPONSE_TRAINING_ONLY,
            "prompt_token_budget": PROMPT_TOKEN_BUDGET,
            "per_device_train_batch_size": PER_DEVICE_TRAIN_BATCH_SIZE,
            "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
            "learning_rate": LEARNING_RATE,
            "max_steps": MAX_STEPS,
            "lora_r": LORA_R,
            "lora_alpha": LORA_ALPHA,
            "packing": PACKING,
            "seed": SEED,
        },
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)

train_result = trainer.train()
trainer.save_model(str(MODEL_OUTPUT_DIR))
tokenizer.save_pretrained(str(MODEL_OUTPUT_DIR))
print(train_result)

# %%
# @title Benchmark fine-tuned model
gc.collect()
torch.cuda.empty_cache()

tuned_metrics = benchmark(test_records, "fine_tuned_model", MAX_EVAL_EXAMPLES_TUNED)
comparison = {"raw": raw_metrics, "fine_tuned": tuned_metrics}
(REPORT_DIR / "benchmark_comparison.json").write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")
print(
    json.dumps(
        {
            "raw": {
                "n": raw_metrics.get("n"),
                "parse_rate": raw_metrics.get("parse_rate"),
                "accuracy_vs_label": raw_metrics.get("accuracy_vs_label"),
                "accuracy_vs_target": raw_metrics.get("accuracy_vs_target"),
            },
            "fine_tuned": {
                "n": tuned_metrics.get("n"),
                "parse_rate": tuned_metrics.get("parse_rate"),
                "accuracy_vs_label": tuned_metrics.get("accuracy_vs_label"),
                "accuracy_vs_target": tuned_metrics.get("accuracy_vs_target"),
            },
        },
        indent=2,
    )
)

# %%
# @title Optional exports
if EXPORT_MERGED_16BIT:
    model.save_pretrained_merged(str(MERGED_OUTPUT_DIR), tokenizer, save_method="merged_16bit")
    print("Saved merged 16-bit model to", MERGED_OUTPUT_DIR)

if EXPORT_GGUF_BF16:
    model.save_pretrained_gguf(str(GGUF_OUTPUT_DIR), tokenizer, quantization_method="bf16")
    print("Saved BF16 GGUF model to", GGUF_OUTPUT_DIR)

# %%
# @title Quick manual inference on one held-out example
if test_records:
    record = test_records[0]
    raw, parsed, parse_mode = generate_one(record)
    print("Label:", record["label"])
    print("Target verdict:", record["target_verdict"])
    print("Parse mode:", parse_mode)
    print("Parsed:", json.dumps(parsed, ensure_ascii=False, indent=2) if parsed else None)
    print("Raw:", raw[:2000])
else:
    print("Quick manual inference skipped because no test rows remained after filtering.")
