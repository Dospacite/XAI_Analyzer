#!/usr/bin/env python3
# %%!
# # Step3-VL-10B Checkpoint Evaluation
#
# Eval-only notebook for the Step3-VL phishing LoRA run.
#
# This notebook:
# - Loads `features/50k.csv`
# - Recreates the deterministic domain-grouped validation/test splits
# - Loads the train-selected feature list from `RUN_DIR/reports/selected_features.json`
# - Loads the latest retained LoRA checkpoint from `RUN_DIR/checkpoints`
# - Patches Step3's generation compatibility issue around `cache_position`
# - Runs cached native `model.generate(...)` evaluation
# - Writes metrics and predictions under `RUN_DIR/reports/checkpoint_eval`
#
# It does not train, and it does not use Step3's visual branch.

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
            "peft",
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

try:
    from google.colab import drive

    drive.mount("/content/drive")
except Exception as exc:
    print(f"Drive mount skipped or unavailable: {exc}")

PROJECT_DIR_STR = "/content/drive/MyDrive/XAI_Analyzer"  # @param {type:"string"}
DATA_CSV_STR = "features/50k.csv"  # @param {type:"string"}
RUN_DIR_STR = "tuning/runs/step3_vl_10b_phishing_lora"  # @param {type:"string"}

PROJECT_DIR = Path(PROJECT_DIR_STR)
DATA_CSV = Path(DATA_CSV_STR)
RUN_DIR = Path(RUN_DIR_STR)

if not DATA_CSV.is_absolute():
    DATA_CSV = PROJECT_DIR / DATA_CSV
if not RUN_DIR.is_absolute():
    RUN_DIR = PROJECT_DIR / RUN_DIR

REPORT_DIR = RUN_DIR / "reports"
CHECKPOINT_DIR = RUN_DIR / "checkpoints"
EVAL_REPORT_DIR = REPORT_DIR / "checkpoint_eval"
EVAL_REPORT_DIR.mkdir(parents=True, exist_ok=True)

print("DATA_CSV:", DATA_CSV)
print("RUN_DIR:", RUN_DIR)
print("CHECKPOINT_DIR:", CHECKPOINT_DIR)
print("EVAL_REPORT_DIR:", EVAL_REPORT_DIR)

# %%!
# @title Imports and evaluation configuration
import gc
import json
import math
import random
import re
import types
from collections import Counter, defaultdict
from typing import Any
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor, StoppingCriteria, StoppingCriteriaList
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.trainer_utils import get_last_checkpoint

try:
    import tldextract
except Exception as exc:
    tldextract = None
    print(f"tldextract unavailable; using urlparse fallback: {exc}")


SEED = 3407  # @param {type:"integer"}
MODEL_ID = "stepfun-ai/Step3-VL-10B"  # @param {type:"string"}
MODEL_REVISION = "5026053b0c2f5dfaa08fc2d149384162c3c8bca1"  # @param {type:"string"}
USE_LATEST_MODEL_REVISION = False  # @param {type:"boolean"}

MAX_SEQ_LENGTH = 16384  # @param {type:"integer"}
MAX_NEW_TOKENS = 768  # @param {type:"integer"}
MAX_EVAL_EXAMPLES = 1000  # @param {type:"integer"}
EVAL_SPLITS = ["validation", "test"]

TRAIN_SIZE = 0.85
VAL_SIZE = 0.05
TEST_SIZE = 0.10

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    print(f"GPU memory free/total GB: {free_bytes / 1e9:.2f}/{total_bytes / 1e9:.2f}")

# %%!
# @title Utility functions
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


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8")


def compact_text(value: Any, max_chars: int = 500) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text[:max_chars].rstrip()


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


def latest_checkpoint(checkpoint_dir: Path) -> Path:
    checkpoint = get_last_checkpoint(str(checkpoint_dir)) if checkpoint_dir.exists() else None
    if not checkpoint:
        raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}")
    return Path(checkpoint)

# %%!
# @title Load data, split, and selected features
if not DATA_CSV.exists():
    raise FileNotFoundError(f"DATA_CSV not found: {DATA_CSV}")

selected_features_path = REPORT_DIR / "selected_features.json"
if not selected_features_path.exists():
    raise FileNotFoundError(f"Selected feature report not found: {selected_features_path}")

selected_feature_report = json.loads(selected_features_path.read_text(encoding="utf-8"))
selected_features = selected_feature_report.get("selected_features") or []
if not selected_features:
    raise ValueError(f"No selected_features found in {selected_features_path}")

df = pd.read_csv(DATA_CSV)
required_columns = {"id", "label", "db", "collection", "url", "final_url", "title"}
missing = sorted(required_columns - set(df.columns))
if missing:
    raise ValueError(f"Missing required columns in {DATA_CSV}: {missing}")

feature_missing = sorted(set(selected_features) - set(df.columns))
if feature_missing:
    raise ValueError(f"Selected features missing from CSV: {feature_missing[:20]}")

df["label"] = df["label"].astype(str)
df["source_diagnostic"] = df["db"].astype(str) + "." + df["collection"].astype(str)
df["registrable_domain"] = df["final_url"].fillna(df["url"]).map(registered_domain)
df["split"] = split_domain_groups(df, SEED)

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
print("Selected feature count:", len(selected_features))

save_json(EVAL_REPORT_DIR / "dataset_summary.json", split_summary)
save_json(EVAL_REPORT_DIR / "selected_features_used.json", {"selected_features": selected_features})

# %%!
# @title Prompt construction
SYSTEM_PROMPT = (
    "You are a text-only phishing website classifier. "
    "Use only the URL, title, and provided engineered feature values. "
    "Do not use or infer from screenshots, images, visual layout, dataset source, collection, hidden labels, or training metadata. "
    "Return strict JSON only, with no Markdown and no chain-of-thought. "
    "Do not include <think>, reasoning, analysis, or explanations outside JSON. "
    "Use this exact schema: "
    '{"verdict":"phishing|benign","confidence_level":"low|medium|high",'
    '"evidence":[{"feature":"feature.name","value":0,"direction":"suspicious|benign|neutral",'
    '"severity":"low|medium|high","statement":"short observable reason"}],'
    '"explanation":"short evidence-grounded explanation"}. '
    "The evidence array must contain at least 3 and at most 7 items. "
    "Every evidence item must cite a feature name that appears in the provided feature_values object."
)


def feature_values_for_row(row: pd.Series) -> dict[str, Any]:
    return {feature: clean_float(row.get(feature)) for feature in selected_features}


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


def make_messages(row: pd.Series) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(make_user_payload(row), ensure_ascii=False, separators=(",", ":")),
                }
            ],
        },
    ]


def strip_visual_placeholders(text: str) -> str:
    forbidden = ["<im_patch>", "<|image_pad|>", "<|vision_start|>", "<|vision_end|>"]
    for token in forbidden:
        if token in text:
            raise ValueError(f"Visual placeholder leaked into text-only prompt: {token}")
    return text

# %%!
# @title Load base model and latest LoRA checkpoint
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

checkpoint_path = latest_checkpoint(CHECKPOINT_DIR)
print("Loading latest checkpoint:", checkpoint_path)

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    revision=revision,
    trust_remote_code=True,
    key_mapping=key_mapping,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
base_model.config.use_cache = True
if GENERATION_EOS_TOKEN_ID is not None and hasattr(base_model, "generation_config"):
    base_model.generation_config.eos_token_id = GENERATION_EOS_TOKEN_ID
    base_model.generation_config.pad_token_id = tokenizer.pad_token_id


def patch_step3_peft_embedding_accessors(model):
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    if not hasattr(base, "model") or not hasattr(base.model, "language_model"):
        return model
    language_model = base.model.language_model
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
    base.get_input_embeddings = peft_get_input_embeddings
    base.set_input_embeddings = peft_set_input_embeddings
    base.model.get_input_embeddings = step3_inner_get_input_embeddings
    base.model.set_input_embeddings = step3_inner_set_input_embeddings
    return model


def patch_step3_prepare_inputs_for_generation(model):
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    if not hasattr(base, "prepare_inputs_for_generation"):
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

    base.prepare_inputs_for_generation = types.MethodType(patched_prepare_inputs_for_generation, base)
    if hasattr(model, "base_model_prepare_inputs_for_generation"):
        model.base_model_prepare_inputs_for_generation = base.prepare_inputs_for_generation
    return model


def patch_step3_forward_for_cached_generation(model):
    """Return past_key_values from Step3 outer forward so generate can cache.

    The remote Step3 outer forward computes logits from the inner model output
    but returns only logits, dropping `past_key_values`. That makes generation
    recompute the whole sequence every token. This patched forward preserves the
    cache fields returned by the inner Qwen3 decoder path.
    """
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    if not hasattr(base, "model") or not hasattr(base, "lm_head"):
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

    base.forward = types.MethodType(patched_forward, base)
    return model


base_model = patch_step3_peft_embedding_accessors(base_model)
base_model = patch_step3_prepare_inputs_for_generation(base_model)
base_model = patch_step3_forward_for_cached_generation(base_model)

model = PeftModel.from_pretrained(base_model, str(checkpoint_path), is_trainable=False)
model = patch_step3_peft_embedding_accessors(model)
model = patch_step3_prepare_inputs_for_generation(model)
model = patch_step3_forward_for_cached_generation(model)
if GENERATION_EOS_TOKEN_ID is not None and hasattr(model, "generation_config"):
    model.generation_config.eos_token_id = GENERATION_EOS_TOKEN_ID
    model.generation_config.pad_token_id = tokenizer.pad_token_id
model.eval()

if hasattr(model, "model"):
    if hasattr(model.model, "vision_model"):
        model.model.vision_model.requires_grad_(False)
    if hasattr(model.model, "vit_large_projector"):
        model.model.vit_large_projector.requires_grad_(False)

print("Loaded checkpoint model.")

# %%!
# @title JSON parsing and evaluation helpers
def render_chat(messages: list[dict[str, Any]], add_generation_prompt: bool = False) -> str:
    return processor.apply_chat_template(messages, add_generation_prompt=add_generation_prompt, tokenize=False)


def render_json_generation_prompt(messages: list[dict[str, Any]]) -> str:
    # Step3's built-in generation prompt starts the assistant with "<think>".
    # For this classifier we want final JSON only, so append a plain assistant turn.
    return render_chat(messages, add_generation_prompt=False) + "<|im_start|>assistant\n"


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


def extract_json_object(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
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
                    except json.JSONDecodeError:
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
    messages = make_messages(row)
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
    validation = validate_payload(payload, set(feature_values_for_row(row).keys()))
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
    for _idx, row in tqdm(frame.iterrows(), total=len(frame), desc=f"checkpoint_eval_{split}"):
        rows.append(generate_for_row(row))
    valid_predictions = [row for row in rows if row["verdict"] in {"phishing", "benign"}]
    metrics = {
        "checkpoint": str(checkpoint_path),
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

# %%!
# @title Run checkpoint evaluation
all_eval = {}
all_predictions = []
for split in EVAL_SPLITS:
    result = evaluate_generation(split, MAX_EVAL_EXAMPLES)
    all_eval[split] = result["metrics"]
    all_predictions.extend(result["predictions"])
    save_json(EVAL_REPORT_DIR / f"{split}_metrics.json", result["metrics"])
    with (EVAL_REPORT_DIR / f"{split}_predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in result["predictions"]:
            handle.write(json.dumps(row, ensure_ascii=False, default=json_default) + "\n")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

save_json(EVAL_REPORT_DIR / "metrics.json", all_eval)
with (EVAL_REPORT_DIR / "predictions.jsonl").open("w", encoding="utf-8") as handle:
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
save_json(EVAL_REPORT_DIR / "evidence_report.json", evidence_report)

print(json.dumps(all_eval, indent=2, ensure_ascii=False))

# %%!
# @title Artifact locations
print("Checkpoint evaluated:", checkpoint_path)
print("Evaluation report directory:", EVAL_REPORT_DIR)
print("Metrics:", EVAL_REPORT_DIR / "metrics.json")
print("Predictions:", EVAL_REPORT_DIR / "predictions.jsonl")

example_prediction = None
if "all_predictions" in globals() and all_predictions:
    example_prediction = all_predictions[0]
elif (EVAL_REPORT_DIR / "predictions.jsonl").exists():
    with (EVAL_REPORT_DIR / "predictions.jsonl").open("r", encoding="utf-8") as handle:
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
