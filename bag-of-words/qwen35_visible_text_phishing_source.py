from __future__ import annotations

# Qwen3.5 0.8B Base visible-text phishing classifier.
#
# This notebook intentionally uses only `input.visible_text` from out_features.jsonl.
# It does not expose URL, title, source, forms, anchors, resources, engineered
# features, or Mongo metadata to the model.

from pathlib import Path
import inspect
import os
import sys

PROJECT_ROOT = Path.cwd()
if PROJECT_ROOT.name == "bag-of-words":
    PROJECT_ROOT = PROJECT_ROOT.parent
elif not (PROJECT_ROOT / "out_features.jsonl").exists():
    for parent in Path.cwd().parents:
        if (parent / "out_features.jsonl").exists():
            PROJECT_ROOT = parent
            break

EXPERIMENT_DIR = PROJECT_ROOT / "bag-of-words"
RUN_DIR = EXPERIMENT_DIR / "runs" / "qwen35_visible_text_phishing"
REPORT_DIR = RUN_DIR / "reports"
ADAPTER_DIR = RUN_DIR / "adapter"

for path in (EXPERIMENT_DIR, RUN_DIR, REPORT_DIR, ADAPTER_DIR):
    path.mkdir(parents=True, exist_ok=True)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

print("Project root:", PROJECT_ROOT)
print("Python executable:", sys.executable)
print("Run directory:", RUN_DIR)
if ".venv" not in sys.executable:
    print("Warning: this notebook is not running from the project .venv.")

# %%!
import importlib.util

required_modules = {
    "accelerate": "accelerate",
    "datasets": "datasets",
    "numpy": "numpy",
    "peft": "peft",
    "sklearn": "scikit-learn",
    "torch": "torch",
    "tqdm": "tqdm",
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
import gc
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from tqdm.auto import tqdm
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    set_seed,
)

try:
    from IPython.display import display
except Exception:
    display = print


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


def env_optional_int(name: str) -> int | None:
    value = os.environ.get(name)
    if value is None or not value.strip() or value.strip().lower() == "auto":
        return None
    return int(value)


def env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    return float(value)


def cuda_total_memory_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return float(torch.cuda.get_device_properties(0).total_memory / (1024**3))


def configure_torch_runtime() -> None:
    if not torch.cuda.is_available():
        return
    allow_tf32 = env_bool("BOW_ALLOW_TF32", True)
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32
    torch.backends.cudnn.benchmark = env_bool("BOW_CUDNN_BENCHMARK", True)
    if allow_tf32 and hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    cuda_backend = getattr(torch.backends, "cuda", None)
    for name in ("enable_flash_sdp", "enable_mem_efficient_sdp", "enable_math_sdp"):
        toggle = getattr(cuda_backend, name, None)
        if callable(toggle):
            toggle(True)


def default_max_seq_length(smoke_test: bool) -> int:
    if smoke_test:
        return 768
    total_gb = cuda_total_memory_gb()
    if not total_gb:
        return 768
    if total_gb < 10:
        return 768
    if total_gb < 16:
        return 1024
    if total_gb < 24:
        return 1536
    if total_gb < 48:
        return 2048
    if total_gb < 80:
        return 3072
    return 4096


def default_train_batch_size(smoke_test: bool) -> int:
    if smoke_test:
        return 1
    total_gb = cuda_total_memory_gb()
    if not total_gb:
        return 1
    if total_gb < 12:
        return 1
    if total_gb < 24:
        return 2
    if total_gb < 48:
        return 4
    if total_gb < 80:
        return 8
    return 16


def default_eval_batch_size(smoke_test: bool) -> int:
    if smoke_test:
        return 2
    total_gb = cuda_total_memory_gb()
    if not total_gb:
        return 1
    if total_gb < 12:
        return 1
    if total_gb < 24:
        return 4
    if total_gb < 48:
        return 8
    if total_gb < 80:
        return 12
    return 16


def default_gradient_accumulation_steps(smoke_test: bool, train_batch_size: int) -> int:
    if smoke_test:
        return 1
    return max(1, 16 // max(1, train_batch_size))


def default_logprob_eval_batch_size(smoke_test: bool) -> int:
    return default_eval_batch_size(smoke_test)


def default_gradient_checkpointing(smoke_test: bool) -> bool:
    if smoke_test:
        return False
    total_gb = cuda_total_memory_gb()
    return bool(total_gb and total_gb < 80)


SEED = env_int("BOW_SEED", 3407)
MODEL_NAME = os.environ.get("BOW_MODEL_NAME", "Qwen/Qwen3.5-0.8B-Base")
DATA_JSONL = Path(os.environ.get("BOW_DATA_JSONL", str(PROJECT_ROOT / "out_features.jsonl")))

configure_torch_runtime()

SMOKE_TEST = env_bool("BOW_SMOKE_TEST", False)
SMOKE_ROWS_PER_LABEL = env_int("BOW_SMOKE_ROWS_PER_LABEL", 256)
MAX_ROWS_PER_LABEL = env_int("BOW_MAX_ROWS_PER_LABEL", 40000)

MAX_VISIBLE_TEXT_CHARS_REQUESTED = env_optional_int("BOW_MAX_VISIBLE_TEXT_CHARS")
MAX_SEQ_LENGTH_REQUESTED = env_optional_int("BOW_MAX_SEQ_LENGTH")
MAX_VISIBLE_TEXT_CHARS = MAX_VISIBLE_TEXT_CHARS_REQUESTED or (512 if SMOKE_TEST else 4000)
MAX_SEQ_LENGTH = MAX_SEQ_LENGTH_REQUESTED or default_max_seq_length(SMOKE_TEST)
DROP_OVERSIZED_EXAMPLES = env_bool("BOW_DROP_OVERSIZED_EXAMPLES", True)
TEXT_SELECTION_STRATEGY = os.environ.get("BOW_TEXT_SELECTION_STRATEGY", "head_tail_terms")
LENGTH_PROFILE_ROWS = env_int("BOW_LENGTH_PROFILE_ROWS", 256 if SMOKE_TEST else 2000)

TEST_SIZE = env_float("BOW_TEST_SIZE", 0.10)
VAL_SIZE = env_float("BOW_VAL_SIZE", 0.05)

LOAD_IN_4BIT = env_bool("BOW_LOAD_IN_4BIT", True)
LORA_R = env_int("BOW_LORA_R", 16)
LORA_ALPHA = env_int("BOW_LORA_ALPHA", 16)
LORA_DROPOUT = env_float("BOW_LORA_DROPOUT", 0.0)
GRADIENT_CHECKPOINTING = env_bool("BOW_GRADIENT_CHECKPOINTING", default_gradient_checkpointing(SMOKE_TEST))

PER_DEVICE_TRAIN_BATCH_SIZE = env_int("BOW_TRAIN_BATCH_SIZE", default_train_batch_size(SMOKE_TEST))
PER_DEVICE_EVAL_BATCH_SIZE = env_int("BOW_EVAL_BATCH_SIZE", default_eval_batch_size(SMOKE_TEST))
GRADIENT_ACCUMULATION_STEPS = env_int(
    "BOW_GRAD_ACCUM_STEPS",
    default_gradient_accumulation_steps(SMOKE_TEST, PER_DEVICE_TRAIN_BATCH_SIZE),
)
LEARNING_RATE = env_float("BOW_LEARNING_RATE", 2e-4)
MAX_STEPS = env_int("BOW_MAX_STEPS", 20 if SMOKE_TEST else 1000)
WARMUP_STEPS = env_int("BOW_WARMUP_STEPS", 2 if SMOKE_TEST else 100)
LOGGING_STEPS = env_int("BOW_LOGGING_STEPS", 5 if SMOKE_TEST else 10)
SAVE_STEPS = env_int("BOW_SAVE_STEPS", 20 if SMOKE_TEST else 250)
DO_EVAL_DURING_TRAINING = env_bool("BOW_EVAL_DURING_TRAINING", False)
DATALOADER_NUM_WORKERS = env_int("BOW_DATALOADER_NUM_WORKERS", 0)
GROUP_BY_LENGTH = env_bool("BOW_GROUP_BY_LENGTH", True)

DO_RAW_BENCHMARK = env_bool("BOW_RAW_BENCHMARK", True)
MAX_EVAL_EXAMPLES_RAW = env_int("BOW_MAX_EVAL_EXAMPLES_RAW", 20 if SMOKE_TEST else 100)
MAX_EVAL_EXAMPLES_TUNED = env_int("BOW_MAX_EVAL_EXAMPLES_TUNED", 50 if SMOKE_TEST else 500)
GEN_MAX_NEW_TOKENS = env_int("BOW_GEN_MAX_NEW_TOKENS", 4)
LOGPROB_EVAL_BATCH_SIZE = env_int("BOW_LOGPROB_EVAL_BATCH_SIZE", default_logprob_eval_batch_size(SMOKE_TEST))
ATTENTION_IMPLEMENTATION = os.environ.get("BOW_ATTENTION_IMPLEMENTATION", "auto").strip().lower()

LABELS = ["benign", "phishing"]

random.seed(SEED)
np.random.seed(SEED)
set_seed(SEED)

print("Model:", MODEL_NAME)
print("Data:", DATA_JSONL)
print("Smoke test:", SMOKE_TEST)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("GPU memory GB:", round(cuda_total_memory_gb(), 2))
print("Max sequence length:", MAX_SEQ_LENGTH)
print("Train batch size:", PER_DEVICE_TRAIN_BATCH_SIZE)
print("Gradient accumulation steps:", GRADIENT_ACCUMULATION_STEPS)
print("Gradient checkpointing:", GRADIENT_CHECKPOINTING)

# %%!
def compact_text(value: Any, max_chars: int | None = None) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if max_chars is not None and len(text) > max_chars:
        return text[:max_chars].rstrip()
    return text


CREDENTIAL_TERMS = [
    "password",
    "login",
    "log in",
    "sign in",
    "signin",
    "verify",
    "verification",
    "account",
    "security",
    "email",
    "username",
    "credential",
    "wallet",
    "payment",
    "bank",
    "invoice",
    "document",
]
CREDENTIAL_PATTERN = re.compile("|".join(re.escape(term) for term in CREDENTIAL_TERMS), re.IGNORECASE)


def quantile(values: list[int | float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * q))))
    return float(ordered[index])


def numeric_profile(values: list[int | float]) -> dict[str, float]:
    return {
        "count": float(len(values)),
        "min": float(min(values)) if values else 0.0,
        "p50": quantile(values, 0.50),
        "p75": quantile(values, 0.75),
        "p90": quantile(values, 0.90),
        "p95": quantile(values, 0.95),
        "p99": quantile(values, 0.99),
        "max": float(max(values)) if values else 0.0,
        "mean": float(np.mean(values)) if values else 0.0,
    }


def make_text_window(text: str, center: int, width: int) -> str:
    half = max(1, width // 2)
    start = max(0, center - half)
    end = min(len(text), center + half)
    return text[start:end].strip()


def select_visible_text(raw_text: str, max_chars: int) -> str:
    text = compact_text(raw_text)
    if len(text) <= max_chars:
        return text
    if TEXT_SELECTION_STRATEGY == "head":
        return text[:max_chars].rstrip()
    if TEXT_SELECTION_STRATEGY == "head_tail":
        head_chars = max(1, int(max_chars * 0.70))
        tail_chars = max_chars - head_chars
        return compact_text(text[:head_chars] + "\n...\n" + text[-tail_chars:], max_chars)

    head_chars = max(1, int(max_chars * 0.45))
    tail_chars = max(1, int(max_chars * 0.20))
    term_budget = max(0, max_chars - head_chars - tail_chars - 16)
    windows = []
    matches = list(CREDENTIAL_PATTERN.finditer(text))
    if matches and term_budget > 0:
        per_window = max(80, term_budget // min(3, len(matches)))
        used_ranges: list[tuple[int, int]] = []
        for match in matches[:8]:
            center = (match.start() + match.end()) // 2
            half = per_window // 2
            span = (max(0, center - half), min(len(text), center + half))
            if any(abs(span[0] - old_start) < 80 for old_start, _ in used_ranges):
                continue
            used_ranges.append(span)
            windows.append(text[span[0] : span[1]].strip())
            if len(compact_text(" ".join(windows))) >= term_budget:
                break
    combined = "\n...\n".join([text[:head_chars].rstrip(), *windows, text[-tail_chars:].lstrip()])
    return compact_text(combined, max_chars)


def normalize_label(value: Any) -> str:
    label = str(value or "").strip().lower()
    if label in {"phishing", "phish", "malicious"}:
        return "phishing"
    if label in {"benign", "legitimate", "safe"}:
        return "benign"
    return ""


def make_visible_text_prompt(visible_text: str) -> str:
    return (
        "Classify the website using only the visible text below. "
        "Answer with exactly one word: phishing or benign.\n\n"
        "Visible text:\n"
        f"{compact_text(visible_text)}"
    )


def iter_jsonl(path: Path, max_rows_per_label: int = 0):
    counts: Counter[str] = Counter()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            label = normalize_label(record.get("label"))
            if label not in set(LABELS):
                continue
            if max_rows_per_label and counts[label] >= max_rows_per_label:
                if all(counts[name] >= max_rows_per_label for name in LABELS):
                    break
                continue
            page_input = record.get("input") or {}
            raw_visible_text = compact_text(page_input.get("visible_text"))
            if not raw_visible_text:
                continue
            counts[label] += 1
            yield {
                "id": str(record.get("id") or ""),
                "label": label,
                "raw_visible_text": raw_visible_text,
            }


def load_visible_text_records() -> list[dict[str, Any]]:
    if not DATA_JSONL.exists():
        raise FileNotFoundError(f"Missing input JSONL: {DATA_JSONL}")
    cap = SMOKE_ROWS_PER_LABEL if SMOKE_TEST else MAX_ROWS_PER_LABEL
    records = list(tqdm(iter_jsonl(DATA_JSONL, cap), desc="load_visible_text"))
    if not records:
        raise RuntimeError(f"No usable visible-text records found in {DATA_JSONL}")
    return records


def raw_text_stats(records_to_profile: list[dict[str, Any]]) -> dict[str, Any]:
    by_label: dict[str, dict[str, Any]] = {}
    for label in LABELS:
        texts = [record["raw_visible_text"] for record in records_to_profile if record["label"] == label]
        by_label[label] = {
            "chars": numeric_profile([len(text) for text in texts]),
            "words": numeric_profile([len(text.split()) for text in texts]),
            "credential_term_match_rate": float(np.mean([bool(CREDENTIAL_PATTERN.search(text)) for text in texts]))
            if texts
            else 0.0,
        }
    all_texts = [record["raw_visible_text"] for record in records_to_profile]
    return {
        "overall": {
            "chars": numeric_profile([len(text) for text in all_texts]),
            "words": numeric_profile([len(text.split()) for text in all_texts]),
            "dataset_visible_text_cap_detected": int(max((len(text) for text in all_texts), default=0)),
        },
        "by_label": by_label,
    }


def balance_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["label"]].append(record)
    if any(not grouped[label] for label in LABELS):
        raise RuntimeError(f"Both labels are required. Counts: {dict(Counter(record['label'] for record in records))}")
    min_count = min(len(grouped[label]) for label in LABELS)
    rng = random.Random(SEED)
    balanced = []
    for label in LABELS:
        items = grouped[label][:]
        rng.shuffle(items)
        balanced.extend(items[:min_count])
    rng.shuffle(balanced)
    return balanced


def stratified_split(
    records: list[dict[str, Any]],
    test_size: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["label"]].append(record)

    left, right = [], []
    for label in LABELS:
        items = grouped[label][:]
        rng.shuffle(items)
        split_count = max(1, int(round(len(items) * test_size))) if test_size > 0 else 0
        split_count = min(split_count, max(0, len(items) - 1))
        right.extend(items[:split_count])
        left.extend(items[split_count:])

    rng.shuffle(left)
    rng.shuffle(right)
    return left, right


records = balance_records(load_visible_text_records())
train_records, test_records = stratified_split(records, TEST_SIZE, SEED)
val_fraction_of_train = VAL_SIZE / max(1e-9, 1.0 - TEST_SIZE)
train_records, val_records = stratified_split(train_records, val_fraction_of_train, SEED + 1)

raw_length_profile = raw_text_stats(records)
print(json.dumps(raw_length_profile, indent=2, ensure_ascii=False))

# %%!
def choose_attention_implementation() -> str | None:
    requested = ATTENTION_IMPLEMENTATION
    if requested in {"", "none", "default"}:
        return None
    if requested != "auto":
        return requested
    if not torch.cuda.is_available():
        return None
    if importlib.util.find_spec("flash_attn") is not None:
        return "flash_attention_2"
    return "sdpa"


def make_model_load_kwargs() -> dict[str, Any]:
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "dtype": dtype,
    }
    if torch.cuda.is_available():
        kwargs["device_map"] = "auto"
    attention_implementation = choose_attention_implementation()
    if attention_implementation:
        kwargs["attn_implementation"] = attention_implementation
    if LOAD_IN_4BIT:
        if importlib.util.find_spec("bitsandbytes") is None:
            raise RuntimeError("BOW_LOAD_IN_4BIT=True requires bitsandbytes.")
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )
    return kwargs


processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer = getattr(processor, "tokenizer", processor)
if getattr(tokenizer, "pad_token", None) is None:
    tokenizer.pad_token = getattr(tokenizer, "eos_token", None)
if hasattr(tokenizer, "padding_side"):
    tokenizer.padding_side = "left"

model_load_kwargs = make_model_load_kwargs()
try:
    model = AutoModelForImageTextToText.from_pretrained(MODEL_NAME, **model_load_kwargs)
except Exception as exc:
    if "attn_implementation" not in model_load_kwargs:
        raise
    failed_attention = model_load_kwargs.pop("attn_implementation")
    print(f"Attention implementation {failed_attention!r} failed during model load; retrying with model default. Error: {exc}")
    model = AutoModelForImageTextToText.from_pretrained(MODEL_NAME, **model_load_kwargs)
if not LOAD_IN_4BIT and torch.cuda.is_available():
    model.to("cuda")

for token_id_name in ("pad_token_id", "eos_token_id", "bos_token_id"):
    token_id = getattr(tokenizer, token_id_name, None)
    if token_id is None:
        continue
    setattr(model.config, token_id_name, token_id)
    if getattr(model, "generation_config", None) is not None:
        setattr(model.generation_config, token_id_name, token_id)

print("Loaded model:", MODEL_NAME)
print("Attention implementation:", model_load_kwargs.get("attn_implementation", "model_default"))
print("Pad token:", getattr(tokenizer, "pad_token", None), getattr(tokenizer, "pad_token_id", None))
print("Vocab size:", getattr(tokenizer, "vocab_size", None))

# %%!
def disable_training_cache(model_to_update: Any) -> None:
    candidates = [
        model_to_update,
        getattr(model_to_update, "base_model", None),
        getattr(getattr(model_to_update, "base_model", None), "model", None),
    ]
    for candidate in candidates:
        config = getattr(candidate, "config", None)
        if hasattr(config, "use_cache"):
            config.use_cache = False
        text_config = getattr(config, "text_config", None)
        if hasattr(text_config, "use_cache"):
            text_config.use_cache = False


def answer_text(label: str) -> str:
    eos = getattr(tokenizer, "eos_token", None) or ""
    return f"{label}{eos}"


def make_messages(prompt: str, answer: str | None = None) -> list[dict[str, Any]]:
    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": prompt}],
        }
    ]
    if answer is not None:
        messages.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": answer}],
            }
        )
    return messages


def apply_template(prompt: str, answer: str | None = None) -> str:
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            make_messages(prompt, answer),
            tokenize=False,
            add_generation_prompt=answer is None,
        )
    if hasattr(processor, "apply_chat_template") and getattr(processor, "chat_template", None):
        return processor.apply_chat_template(
            make_messages(prompt, answer),
            tokenize=False,
            add_generation_prompt=answer is None,
        )
    if answer is None:
        return f"User: {prompt}\nAssistant:"
    return f"User: {prompt}\nAssistant: {answer}"


def encode_text(text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False)
    return list(encoded["input_ids"])


def gpu_auto_max_seq_length() -> int:
    return default_max_seq_length(SMOKE_TEST)


def candidate_char_caps(max_observed: int) -> list[int]:
    caps = [512, 768, 1024, 1536, 2048, 3072, 4000, 6000, 8000]
    return sorted({cap for cap in caps if cap <= max(512, max_observed)})


def token_length_for_record(record: dict[str, Any], char_cap: int) -> int:
    visible_text = select_visible_text(record["raw_visible_text"], char_cap)
    prompt = make_visible_text_prompt(visible_text)
    full_text = apply_template(prompt, answer=answer_text(record["label"]))
    return len(encode_text(full_text))


def profile_token_budget(records_to_profile: list[dict[str, Any]]) -> tuple[int, int, dict[str, Any]]:
    auto_seq_length = MAX_SEQ_LENGTH_REQUESTED or gpu_auto_max_seq_length()
    observed_max_chars = int(raw_length_profile["overall"]["chars"]["max"])
    sample = records_to_profile[:LENGTH_PROFILE_ROWS]
    if not sample:
        raise RuntimeError("No rows available for token-length profiling.")

    caps = [MAX_VISIBLE_TEXT_CHARS_REQUESTED] if MAX_VISIBLE_TEXT_CHARS_REQUESTED else candidate_char_caps(observed_max_chars)
    caps = [int(cap) for cap in caps if cap]
    candidate_rows = []
    for cap in tqdm(caps, desc="profile_length_candidates"):
        lengths = [token_length_for_record(record, cap) for record in sample]
        candidate_rows.append(
            {
                "char_cap": cap,
                "token_lengths": numeric_profile(lengths),
                "estimated_drop_rate_at_max_seq": float(np.mean([length > auto_seq_length for length in lengths])),
                "fits_p95": bool(quantile(lengths, 0.95) <= auto_seq_length),
                "fits_p99": bool(quantile(lengths, 0.99) <= auto_seq_length),
            }
        )

    fitting = [row for row in candidate_rows if row["fits_p95"] and row["estimated_drop_rate_at_max_seq"] <= 0.05]
    chosen_cap = int((fitting[-1] if fitting else candidate_rows[0])["char_cap"])
    if MAX_VISIBLE_TEXT_CHARS_REQUESTED:
        chosen_cap = int(MAX_VISIBLE_TEXT_CHARS_REQUESTED)
    profile = {
        "profile_rows": len(sample),
        "gpu_auto_max_seq_length": gpu_auto_max_seq_length(),
        "chosen_max_seq_length": auto_seq_length,
        "chosen_max_visible_text_chars": chosen_cap,
        "text_selection_strategy": TEXT_SELECTION_STRATEGY,
        "candidates": candidate_rows,
        "note": (
            "The current out_features.jsonl appears to have visible_text pre-capped at "
            f"{observed_max_chars} chars; this notebook cannot recover text beyond that source cap."
        ),
    }
    return chosen_cap, auto_seq_length, profile


MAX_VISIBLE_TEXT_CHARS, MAX_SEQ_LENGTH, token_length_profile = profile_token_budget(train_records)
for split_records in (train_records, val_records, test_records):
    for record in split_records:
        record["visible_text"] = select_visible_text(record["raw_visible_text"], MAX_VISIBLE_TEXT_CHARS)

dataset_summary = {
    "model_name": MODEL_NAME,
    "data_jsonl": str(DATA_JSONL),
    "model_visible_fields": ["input.visible_text"],
    "excluded_model_visible_fields": [
        "id",
        "label",
        "source",
        "url",
        "final_url",
        "title",
        "meta",
        "forms",
        "anchors",
        "iframes",
        "resources",
        "measurements",
        "features",
    ],
    "smoke_test": SMOKE_TEST,
    "cuda_total_memory_gb": cuda_total_memory_gb(),
    "rows": len(records),
    "label_counts": dict(Counter(record["label"] for record in records)),
    "train_rows": len(train_records),
    "validation_rows": len(val_records),
    "test_rows": len(test_records),
    "train_label_counts": dict(Counter(record["label"] for record in train_records)),
    "validation_label_counts": dict(Counter(record["label"] for record in val_records)),
    "test_label_counts": dict(Counter(record["label"] for record in test_records)),
    "raw_length_profile": raw_length_profile,
    "token_length_profile": token_length_profile,
    "max_visible_text_chars": MAX_VISIBLE_TEXT_CHARS,
    "max_seq_length": MAX_SEQ_LENGTH,
}
(RUN_DIR / "dataset_summary.json").write_text(
    json.dumps(dataset_summary, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

print(json.dumps(dataset_summary, indent=2, ensure_ascii=False))
display(
    {
        "sample_label": train_records[0]["label"],
        "sample_prompt": make_visible_text_prompt(train_records[0]["visible_text"])[:1200],
    }
)


def longest_common_prefix_length(left: list[int], right: list[int]) -> int:
    prefix = 0
    for a, b in zip(left, right):
        if a != b:
            break
        prefix += 1
    return prefix


def left_truncate_to_max_length(input_ids: list[int], labels: list[int], max_length: int) -> tuple[list[int], list[int]]:
    if len(input_ids) <= max_length:
        return input_ids, labels
    start = len(input_ids) - max_length
    return input_ids[start:], labels[start:]


def tokenize_record(record: dict[str, Any]) -> dict[str, Any] | None:
    prompt = make_visible_text_prompt(record["visible_text"])
    prompt_text = apply_template(prompt, answer=None)
    full_text = apply_template(prompt, answer=answer_text(record["label"]))
    prompt_ids = encode_text(prompt_text)
    full_ids = encode_text(full_text)

    prompt_len = len(prompt_ids)
    if prompt_len > len(full_ids) or full_ids[:prompt_len] != prompt_ids:
        prompt_len = longest_common_prefix_length(prompt_ids, full_ids)

    if len(full_ids) > MAX_SEQ_LENGTH:
        if DROP_OVERSIZED_EXAMPLES:
            return None

    labels = full_ids[:]
    for index in range(min(prompt_len, len(labels))):
        labels[index] = -100
    full_ids, labels = left_truncate_to_max_length(full_ids, labels, MAX_SEQ_LENGTH)
    if all(label == -100 for label in labels):
        return None

    return {
        "id": record["id"],
        "label_name": record["label"],
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
        "prompt_text": prompt_text,
    }


def build_tokenized_dataset(records_to_tokenize: list[dict[str, Any]], split_name: str) -> tuple[Dataset, list[dict[str, Any]]]:
    rows = []
    kept_records = []
    dropped = 0
    for record in tqdm(records_to_tokenize, desc=f"tokenize_{split_name}"):
        row = tokenize_record(record)
        if row is None:
            dropped += 1
            continue
        rows.append({key: row[key] for key in ("input_ids", "attention_mask", "labels")})
        kept_records.append(record)
    if not rows:
        raise RuntimeError(f"No tokenized rows remained for {split_name}. Increase MAX_SEQ_LENGTH or reduce visible text.")
    print(f"{split_name}: kept={len(rows)} dropped={dropped}")
    return Dataset.from_list(rows), kept_records


train_dataset, train_records = build_tokenized_dataset(train_records, "train")
val_dataset, val_records = build_tokenized_dataset(val_records, "validation")
test_dataset, test_records = build_tokenized_dataset(test_records, "test")

print(train_dataset)
print(val_dataset)

# %%!
@dataclass
class CausalClassificationCollator:
    tokenizer: Any

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        label_sequences = [feature.pop("labels") for feature in features]
        batch = self.tokenizer.pad(
            features,
            padding=True,
            return_tensors="pt",
        )
        max_length = batch["input_ids"].shape[1]
        padded_labels = []
        for sequence in label_sequences:
            pad_len = max_length - len(sequence)
            if getattr(self.tokenizer, "padding_side", "right") == "left":
                padded = [-100] * pad_len + sequence
            else:
                padded = sequence + [-100] * pad_len
            padded_labels.append(padded)
        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        return batch


def model_input_device() -> torch.device:
    device = getattr(model, "device", None)
    if isinstance(device, torch.device) and device.type != "meta":
        return device
    for parameter in model.parameters():
        if parameter.device.type != "meta":
            return parameter.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def normalize_prediction_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z]+", " ", text)
    tokens = text.split()
    for token in tokens[:10]:
        if token in {"phishing", "phish"}:
            return "phishing"
        if token in {"benign", "legitimate", "safe"}:
            return "benign"
    if "phishing" in text:
        return "phishing"
    if "benign" in text:
        return "benign"
    return "unparseable"


@torch.inference_mode()
def score_label_candidate_batch(records_to_eval: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not records_to_eval:
        return []
    model.eval()
    candidate_features: list[dict[str, Any]] = []
    candidate_meta: list[tuple[int, str]] = []
    scores_by_record: list[dict[str, float]] = [dict.fromkeys(LABELS, float("-inf")) for _ in records_to_eval]

    for record_index, record in enumerate(records_to_eval):
        prompt = make_visible_text_prompt(record["visible_text"])
        prompt_text = apply_template(prompt, answer=None)
        prompt_ids = encode_text(prompt_text)
        for candidate in LABELS:
            full_ids = encode_text(apply_template(prompt, answer=answer_text(candidate)))
            prompt_len = len(prompt_ids)
            if prompt_len > len(full_ids) or full_ids[:prompt_len] != prompt_ids:
                prompt_len = longest_common_prefix_length(prompt_ids, full_ids)
            labels = full_ids[:]
            for index in range(min(prompt_len, len(labels))):
                labels[index] = -100
            full_ids, labels = left_truncate_to_max_length(full_ids, labels, MAX_SEQ_LENGTH)
            candidate_features.append(
                {
                    "input_ids": full_ids,
                    "attention_mask": [1] * len(full_ids),
                    "labels": labels,
                }
            )
            candidate_meta.append((record_index, candidate))

    if not candidate_features:
        return []
    device = model_input_device()
    batch = CausalClassificationCollator(tokenizer=tokenizer)(candidate_features)
    labels = batch.pop("labels")
    batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
    labels = labels.to(device, non_blocking=True)
    outputs = model(**batch)
    shift_logits = outputs.logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    token_losses = F.cross_entropy(
        shift_logits.transpose(1, 2),
        shift_labels,
        ignore_index=-100,
        reduction="none",
    )
    token_counts = (shift_labels != -100).sum(dim=1)
    sequence_scores = -(token_losses.sum(dim=1) / token_counts.clamp_min(1))

    for index, (record_index, candidate) in enumerate(candidate_meta):
        if int(token_counts[index].item()) == 0:
            continue
        scores_by_record[record_index][candidate] = float(sequence_scores[index].item())

    rows: list[dict[str, Any]] = []
    for record, scores in zip(records_to_eval, scores_by_record):
        prediction = max(scores, key=scores.get)
        rows.append(
            {
                "id": record["id"],
                "label": record["label"],
                "prediction": prediction,
                "label_logprobs": scores,
                "margin": float(scores["phishing"] - scores["benign"]),
            }
        )
    return rows


def score_label_candidates(record: dict[str, Any]) -> dict[str, Any]:
    return score_label_candidate_batch([record])[0]


def score_batch(records_to_eval: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return score_label_candidate_batch(records_to_eval)


@torch.inference_mode()
def generate_batch(records_to_eval: list[dict[str, Any]]) -> list[dict[str, Any]]:
    model.eval()
    prompts = [apply_template(make_visible_text_prompt(record["visible_text"]), answer=None) for record in records_to_eval]
    encoded = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        return_tensors="pt",
    )
    encoded = {key: value.to(model_input_device()) for key, value in encoded.items()}
    prompt_width = encoded["input_ids"].shape[-1]
    generated = model.generate(
        **encoded,
        max_new_tokens=GEN_MAX_NEW_TOKENS,
        do_sample=False,
        pad_token_id=getattr(tokenizer, "pad_token_id", None),
        eos_token_id=getattr(tokenizer, "eos_token_id", None),
    )
    rows = []
    for record, output in zip(records_to_eval, generated):
        raw = tokenizer.decode(output[prompt_width:], skip_special_tokens=True)
        rows.append(
            {
                "id": record["id"],
                "label": record["label"],
                "prediction": normalize_prediction_text(raw),
                "raw_output": raw,
            }
        )
    return rows


def make_report_rows(records_to_eval: list[dict[str, Any]], limit: int, name: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    sample = records_to_eval[:limit] if limit else records_to_eval
    rows: list[dict[str, Any]] = []
    for start in tqdm(range(0, len(sample), LOGPROB_EVAL_BATCH_SIZE), desc=f"score_{name}"):
        rows.extend(score_batch(sample[start : start + LOGPROB_EVAL_BATCH_SIZE]))

    y_true = [row["label"] for row in rows]
    y_pred = [row["prediction"] for row in rows]
    report_labels = LABELS
    metrics = {
        "name": name,
        "evaluation_method": "mean_label_log_probability",
        "n": len(rows),
        "accuracy": accuracy_score(y_true, y_pred) if rows else 0.0,
        "macro_f1": f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0) if rows else 0.0,
        "weighted_f1": f1_score(y_true, y_pred, labels=LABELS, average="weighted", zero_division=0) if rows else 0.0,
        "prediction_counts": dict(Counter(y_pred)),
        "confusion_labels": report_labels,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=report_labels).tolist() if rows else [],
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=report_labels,
            output_dict=True,
            zero_division=0,
        )
        if rows
        else {},
    }
    return rows, metrics


def save_benchmark(name: str, rows: list[dict[str, Any]], metrics: dict[str, Any]) -> None:
    (REPORT_DIR / f"{name}_predictions.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )
    (REPORT_DIR / f"{name}_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps({key: metrics[key] for key in ("name", "n", "accuracy", "macro_f1", "weighted_f1")}, indent=2))

# %%!
if DO_RAW_BENCHMARK and MAX_EVAL_EXAMPLES_RAW:
    raw_rows, raw_metrics = make_report_rows(test_records, MAX_EVAL_EXAMPLES_RAW, "raw_model")
    save_benchmark("raw_model", raw_rows, raw_metrics)
else:
    raw_metrics = {"name": "raw_model", "n": 0, "skipped": True}

# %%!
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

if LOAD_IN_4BIT:
    disable_training_cache(model)
    prepare_kwargs: dict[str, Any] = {}
    prepare_params = inspect.signature(prepare_model_for_kbit_training).parameters
    if "use_gradient_checkpointing" in prepare_params:
        prepare_kwargs["use_gradient_checkpointing"] = GRADIENT_CHECKPOINTING
    if GRADIENT_CHECKPOINTING and "gradient_checkpointing_kwargs" in prepare_params:
        prepare_kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": False}
    model = prepare_model_for_kbit_training(model, **prepare_kwargs)

lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules="all-linear",
)
model = get_peft_model(model, lora_config)
disable_training_cache(model)
model.print_trainable_parameters()

# %%!
def training_arguments_kwargs() -> dict[str, Any]:
    import inspect

    params = inspect.signature(TrainingArguments.__init__).parameters
    kwargs: dict[str, Any] = {
        "output_dir": str(ADAPTER_DIR),
        "per_device_train_batch_size": PER_DEVICE_TRAIN_BATCH_SIZE,
        "per_device_eval_batch_size": PER_DEVICE_EVAL_BATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "learning_rate": LEARNING_RATE,
        "max_steps": MAX_STEPS,
        "warmup_steps": WARMUP_STEPS,
        "logging_steps": LOGGING_STEPS,
        "save_steps": SAVE_STEPS,
        "save_total_limit": 2,
        "save_strategy": "steps",
        "report_to": "none",
        "seed": SEED,
        "data_seed": SEED,
        "remove_unused_columns": False,
        "optim": "paged_adamw_8bit" if LOAD_IN_4BIT else "adamw_torch",
        "weight_decay": 0.001,
        "lr_scheduler_type": "linear",
    }
    if torch.cuda.is_available():
        kwargs["dataloader_pin_memory"] = True
    if "dataloader_num_workers" in params:
        kwargs["dataloader_num_workers"] = DATALOADER_NUM_WORKERS
    if "group_by_length" in params:
        kwargs["group_by_length"] = GROUP_BY_LENGTH
    if "gradient_checkpointing" in params:
        kwargs["gradient_checkpointing"] = GRADIENT_CHECKPOINTING
    if "tf32" in params:
        kwargs["tf32"] = env_bool("BOW_ALLOW_TF32", True) and torch.cuda.is_available()
    eval_value = "steps" if DO_EVAL_DURING_TRAINING else "no"
    if "eval_strategy" in params:
        kwargs["eval_strategy"] = eval_value
    else:
        kwargs["evaluation_strategy"] = eval_value
    if DO_EVAL_DURING_TRAINING:
        kwargs["eval_steps"] = SAVE_STEPS
    if "bf16" in params:
        kwargs["bf16"] = bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    if "fp16" in params:
        kwargs["fp16"] = bool(torch.cuda.is_available() and not torch.cuda.is_bf16_supported())
    return kwargs


training_config = {
    "model_name": MODEL_NAME,
    "base_model_kind": "Qwen3.5 0.8B Base",
    "task": "visible_text_binary_phishing_classification",
    "model_visible_fields": ["input.visible_text"],
    "evaluation_method": "mean_label_log_probability",
    "load_in_4bit": LOAD_IN_4BIT,
    "cuda_total_memory_gb": cuda_total_memory_gb(),
    "requested_max_visible_text_chars": MAX_VISIBLE_TEXT_CHARS_REQUESTED,
    "requested_max_seq_length": MAX_SEQ_LENGTH_REQUESTED,
    "max_visible_text_chars": MAX_VISIBLE_TEXT_CHARS,
    "max_seq_length": MAX_SEQ_LENGTH,
    "text_selection_strategy": TEXT_SELECTION_STRATEGY,
    "drop_oversized_examples": DROP_OVERSIZED_EXAMPLES,
    "test_size": TEST_SIZE,
    "val_size": VAL_SIZE,
    "lora_r": LORA_R,
    "lora_alpha": LORA_ALPHA,
    "lora_dropout": LORA_DROPOUT,
    "gradient_checkpointing": GRADIENT_CHECKPOINTING,
    "per_device_train_batch_size": PER_DEVICE_TRAIN_BATCH_SIZE,
    "per_device_eval_batch_size": PER_DEVICE_EVAL_BATCH_SIZE,
    "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
    "logprob_eval_batch_size": LOGPROB_EVAL_BATCH_SIZE,
    "attention_implementation": model_load_kwargs.get("attn_implementation", "model_default"),
    "allow_tf32": env_bool("BOW_ALLOW_TF32", True),
    "group_by_length": GROUP_BY_LENGTH,
    "dataloader_num_workers": DATALOADER_NUM_WORKERS,
    "effective_train_batch_size": PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS,
    "learning_rate": LEARNING_RATE,
    "max_steps": MAX_STEPS,
    "warmup_steps": WARMUP_STEPS,
    "raw_length_profile": raw_length_profile,
    "token_length_profile": token_length_profile,
    "seed": SEED,
}
(RUN_DIR / "training_config.json").write_text(
    json.dumps(training_config, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

trainer_kwargs = {
    "model": model,
    "args": TrainingArguments(**training_arguments_kwargs()),
    "train_dataset": train_dataset,
    "eval_dataset": val_dataset if DO_EVAL_DURING_TRAINING else None,
    "data_collator": CausalClassificationCollator(tokenizer=tokenizer),
}
try:
    trainer = Trainer(processing_class=processor, **trainer_kwargs)
except TypeError:
    trainer = Trainer(tokenizer=tokenizer, **trainer_kwargs)

train_result = trainer.train()
trainer.save_model(str(ADAPTER_DIR))
processor.save_pretrained(str(ADAPTER_DIR))
print(train_result)

# %%!
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

tuned_rows, tuned_metrics = make_report_rows(test_records, MAX_EVAL_EXAMPLES_TUNED, "fine_tuned_model")
save_benchmark("fine_tuned_model", tuned_rows, tuned_metrics)

comparison = {
    "raw": raw_metrics,
    "fine_tuned": tuned_metrics,
}
(REPORT_DIR / "benchmark_comparison.json").write_text(
    json.dumps(comparison, indent=2, ensure_ascii=False),
    encoding="utf-8",
)
print(
    json.dumps(
        {
            "raw": {
                "n": raw_metrics.get("n"),
                "accuracy": raw_metrics.get("accuracy"),
                "macro_f1": raw_metrics.get("macro_f1"),
            },
            "fine_tuned": {
                "n": tuned_metrics.get("n"),
                "accuracy": tuned_metrics.get("accuracy"),
                "macro_f1": tuned_metrics.get("macro_f1"),
            },
        },
        indent=2,
    )
)

# %%!
record = test_records[0]
manual = score_label_candidates(record)
print("Expected:", record["label"])
print("Prediction:", manual["prediction"])
print("Label logprobs:", json.dumps(manual["label_logprobs"], indent=2))
