#!/usr/bin/env python3
"""Convert Stage B phishing JSONL into a local Hugging Face DatasetDict.

Input is the JSONL produced by extract_features_jsonl.py:
  {"id": "...", "label": "phishing|benign", "source": "...", "input": {...}, "features": [...]}

The saved dataset keeps label/source as metadata columns, but the model-visible
prompt never includes them.
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - local convenience fallback
    tqdm = None


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("out_features.jsonl"), help="Stage B JSONL input path.")
    parser.add_argument("--output", type=Path, default=Path("hf_phishing_dataset"), help="Output directory for save_to_disk.")
    parser.add_argument("--max-records", type=int, default=0, help="Read at most this many JSONL rows. 0 means all.")
    parser.add_argument("--max-train-examples", type=int, default=0, help="Cap train split after splitting. 0 means no cap.")
    parser.add_argument("--test-size", type=float, default=0.10, help="Fraction reserved for test.")
    parser.add_argument("--val-size", type=float, default=0.05, help="Fraction reserved for validation from the whole dataset.")
    parser.add_argument("--seed", type=int, default=3407, help="Random seed for balancing and splits.")
    parser.add_argument("--max-evidence-items", type=int, default=4, help="Maximum evidence items in target JSON.")
    parser.add_argument("--no-balance", action="store_true", help="Do not downsample to equal phishing/benign counts.")
    parser.add_argument("--no-gemma4-text", action="store_true", help="Do not add manually templated Gemma 4 text column.")
    parser.add_argument("--include-messages-object", action="store_true", help="Also store nested messages objects; off by default to reduce memory.")
    parser.add_argument("--keep-intermediate-jsonl", action="store_true", help="Keep converted split JSONL files next to the output dataset.")
    parser.add_argument("--no-progress", action="store_true", help="Disable progress bars.")
    parser.add_argument("--dry-run", action="store_true", help="Build and summarize rows, but do not import datasets or save.")
    return parser.parse_args()


def progress(iterable, **kwargs):
    if tqdm is None or kwargs.pop("disable", False):
        return iterable
    return tqdm(iterable, **kwargs)


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
    """Keep model-visible website observation compact and source/label free."""
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
        "visible_text": compact_text(page_input.get("visible_text"), 2500),
        "forms": {
            "total_observed": collection_total(forms),
            "items": redact_large_value(collection_items(forms)[:20], 240),
        },
        "anchors": {
            "total_observed": collection_total(anchors),
            "items": redact_large_value(collection_items(anchors)[:30], 200),
        },
        "iframes": {
            "total_observed": collection_total(iframes),
            "items": redact_large_value(collection_items(iframes)[:10], 200),
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
        "id": str(feature.get("id") or ""),
        "direction": str(feature.get("direction") or ""),
        "severity": str(feature.get("severity") or ""),
        "value": value,
        "statement": compact_text(feature.get("statement"), 240),
    }


def select_target_evidence(label: str, features: list[dict[str, Any]], max_items: int) -> list[dict[str, Any]]:
    verdict = "phishing" if label == "phishing" else "benign" if label == "benign" else "uncertain"
    wanted_direction = "suspicious" if verdict == "phishing" else "benign" if verdict == "benign" else "uncertain"
    candidates = []
    for feature in features or []:
        if not isinstance(feature, dict):
            continue
        feature_id = feature.get("id")
        if feature_id in EXCLUDED_TARGET_FEATURE_IDS or feature_id not in TARGETABLE_FEATURE_IDS:
            continue
        if feature.get("direction") != wanted_direction:
            continue
        if feature_id == "form.external_action":
            relationship = (feature.get("value") or {}).get("relationship")
            if relationship not in {"unrelated_third_party", "unknown"}:
                continue
        candidates.append(feature)

    candidates.sort(
        key=lambda feature: (
            SEVERITY_RANK.get(str(feature.get("severity")), 0),
            1 if not (feature.get("supervision") or {}).get("primary_eligible") else 2,
            str(feature.get("id")),
        ),
        reverse=True,
    )
    return [sanitize_feature_for_target(feature) for feature in candidates[:max_items]]


def confidence_from_evidence(label: str, evidence: list[dict[str, Any]]) -> str:
    if label not in {"phishing", "benign"}:
        return "low"
    if any(item.get("severity") == "high" for item in evidence) and len(evidence) >= 2:
        return "high"
    if evidence:
        return "medium"
    return "low"


def make_target_output(record: dict[str, Any], max_evidence_items: int) -> dict[str, Any]:
    label = str(record.get("label") or "uncertain")
    verdict = "phishing" if label == "phishing" else "benign" if label == "benign" else "uncertain"
    evidence = select_target_evidence(label, record.get("features") or [], max_evidence_items)
    return {
        "verdict": verdict,
        "confidence_level": confidence_from_evidence(label, evidence),
        "evidence": evidence,
    }


def make_prompt(record: dict[str, Any]) -> str:
    payload = {
        "task": "Classify the website as phishing, benign, or uncertain and return strict JSON.",
        "website_observation": prune_input(record.get("input") or {}),
    }
    return f"{SYSTEM_PROMPT}\n\nWebsite observation:\n{json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}"


def make_messages(prompt: str, response: str) -> list[dict[str, Any]]:
    return [
        {"role": "user", "content": [{"type": "text", "text": prompt}]},
        {"role": "assistant", "content": [{"type": "text", "text": response}]},
    ]


def make_gemma4_text(prompt: str, response: str) -> str:
    return f"<|turn>user\n{prompt}<turn|>\n<|turn>model\n{response}<turn|>"


def iter_jsonl(path: Path, max_records: int = 0):
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if max_records and index >= max_records:
                break
            line = line.strip()
            if line:
                yield json.loads(line)


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


def load_records(args: argparse.Namespace) -> list[dict[str, Any]]:
    records = []
    iterator = iter_jsonl(args.input, args.max_records)
    for record in progress(iterator, desc="load_jsonl", disable=args.no_progress):
        if record.get("label") not in {"phishing", "benign"}:
            continue
        if not record.get("input"):
            continue
        records.append(record)
    if not records:
        raise SystemExit(f"No usable records found in {args.input}")
    return records


def balance_records(records: list[dict[str, Any]], seed: int, enabled: bool) -> list[dict[str, Any]]:
    if not enabled:
        return records[:]
    rng = random.Random(seed)
    by_label = defaultdict(list)
    for record in records:
        by_label[record["label"]].append(record)
    min_count = min(len(by_label["phishing"]), len(by_label["benign"]))
    if min_count == 0:
        raise SystemExit(f"Cannot balance labels with counts: {dict(Counter(record['label'] for record in records))}")
    balanced = []
    for label in ("phishing", "benign"):
        items = by_label[label][:]
        rng.shuffle(items)
        balanced.extend(items[:min_count])
    rng.shuffle(balanced)
    return balanced


def split_records(args: argparse.Namespace, records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    train, test = stratified_split(records, args.test_size, args.seed)
    val_fraction_of_train = args.val_size / max(1e-9, 1.0 - args.test_size)
    train, validation = stratified_split(train, val_fraction_of_train, args.seed + 1)
    if args.max_train_examples and len(train) > args.max_train_examples:
        train = train[: args.max_train_examples]
    return {"train": train, "validation": validation, "test": test}


def convert_record(record: dict[str, Any], split: str, args: argparse.Namespace) -> dict[str, Any]:
    prompt = make_prompt(record)
    target = make_target_output(record, args.max_evidence_items)
    response = json.dumps(target, ensure_ascii=False, separators=(",", ":"))
    messages = make_messages(prompt, response)
    row = {
        "id": str(record.get("id") or ""),
        "label": str(record.get("label") or ""),
        "source": str(record.get("source") or ""),
        "split": split,
        "prompt": prompt,
        "response": response,
        "messages_json": json.dumps(messages, ensure_ascii=False, separators=(",", ":")),
        "target_json": response,
        "target_feature_ids": [item.get("id", "") for item in target["evidence"]],
    }
    if args.include_messages_object:
        row["messages"] = messages
    if not args.no_gemma4_text:
        row["text"] = make_gemma4_text(prompt, response)
    return row


def summarize_splits(splits: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    return {
        split: {"count": len(records), "labels": dict(Counter(record["label"] for record in records))}
        for split, records in splits.items()
    }


def save_dataset(splits: dict[str, list[dict[str, Any]]], args: argparse.Namespace) -> None:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("Install Hugging Face datasets first: pip install datasets") from exc

    args.output.mkdir(parents=True, exist_ok=True)
    intermediate_dir = args.output.with_name(args.output.name + "_jsonl")
    intermediate_dir.mkdir(parents=True, exist_ok=True)
    data_files = {}
    for split, records in splits.items():
        split_path = intermediate_dir / f"{split}.jsonl"
        with split_path.open("w", encoding="utf-8") as handle:
            for record in progress(records, desc=f"convert_{split}", disable=args.no_progress):
                row = convert_record(record, split, args)
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        data_files[split] = str(split_path)
        gc.collect()

    dataset_dict = load_dataset("json", data_files=data_files)
    dataset_dict.save_to_disk(str(args.output))
    metadata = {
        "input": str(args.input),
        "output": str(args.output),
        "intermediate_jsonl_dir": str(intermediate_dir),
        "balanced": not args.no_balance,
        "summary": summarize_splits(splits),
        "columns": list(dataset_dict["train"].column_names) if len(dataset_dict["train"]) else [],
        "notes": [
            "label and source are metadata columns.",
            "prompt/messages/text do not include label or source.",
            "For Gemma 4 training, prefer applying the tokenizer chat template to messages when possible.",
        ],
    }
    (args.output / "conversion_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    if not args.keep_intermediate_jsonl:
        for split_path in intermediate_dir.glob("*.jsonl"):
            split_path.unlink()
        try:
            intermediate_dir.rmdir()
        except OSError:
            pass


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    records = load_records(args)
    print("Loaded:", len(records), dict(Counter(record["label"] for record in records)))
    records = balance_records(records, args.seed, enabled=not args.no_balance)
    print("After balancing:", len(records), dict(Counter(record["label"] for record in records)))
    splits = split_records(args, records)
    print("Splits:", json.dumps(summarize_splits(splits), ensure_ascii=False, indent=2))
    if args.dry_run:
        sample = convert_record(splits["train"][0], "train", args)
        print("Sample row keys:", list(sample.keys()))
        print("Sample prompt:", sample["prompt"][:1000])
        print("Sample response:", sample["response"][:1000])
        return
    save_dataset(splits, args)
    print(f"Saved Hugging Face DatasetDict to {args.output}")


if __name__ == "__main__":
    main()
