#!/usr/bin/env python3
"""Audit Stage B feature JSONL for shortcut learning and bias risk.

The script reads output from extract_features_jsonl.py and reports:
- label/source distributions and source-label leakage
- feature prevalence, entropy, mutual information, and purity by label/source
- measurement threshold shortcuts
- categorical feature value shortcuts
- highly redundant feature pairs

It does not modify the dataset.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Iterable
from urllib.parse import urlparse

from tqdm import tqdm


SENSITIVE_INPUT_KEYS = {
    "source",
    "verdict",
    "expected_output",
    "features",
    "evidence",
    "phishing",
    "benign",
    "collection",
    "database",
    "db",
    "dataset_label",
    "target_label",
    "class_label",
    "is_phishing",
}

SKIP_VALUE_KEYS = {
    "url",
    "url_sample",
    "title",
    "snippet",
    "contact",
    "domain",
    "page_domain",
    "href_domain",
    "action_domain",
    "favicon_domain",
    "hostname",
    "final_domain",
    "start_domain",
    "claimed_brand_known_domains",
    "checked_terms",
    "examples",
}

VALUE_ATOM_KEYS = {
    "relationship",
    "provider_relationship",
    "match_type",
    "target_brand",
    "matched_alias",
    "matched_label",
    "claimed_brand",
    "matched",
    "favicon_brand",
    "page_brand",
    "kind",
    "claim_provenance",
    "scheme",
    "terms",
    "names",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze Stage B feature JSONL for shortcut learning, entropy, and bias."
    )
    parser.add_argument("--input", type=Path, required=True, help="Stage B JSONL file, or '-' for stdin.")
    parser.add_argument("--output", type=Path, default=None, help="Markdown report path. Defaults to stdout.")
    parser.add_argument("--json-output", type=Path, default=None, help="Optional machine-readable JSON summary path.")
    parser.add_argument("--max-records", type=int, default=0, help="Analyze at most this many records. 0 means all.")
    parser.add_argument("--min-count", type=int, default=20, help="Minimum support for shortcut tables.")
    parser.add_argument("--top", type=int, default=40, help="Rows per report table.")
    parser.add_argument("--high-purity", type=float, default=0.95, help="Purity threshold for high shortcut risk.")
    parser.add_argument("--medium-purity", type=float, default=0.85, help="Purity threshold for medium shortcut risk.")
    parser.add_argument("--high-nmi", type=float, default=0.25, help="Normalized mutual information threshold for high shortcut risk.")
    parser.add_argument("--medium-nmi", type=float, default=0.10, help="Normalized mutual information threshold for medium shortcut risk.")
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bar.")
    return parser.parse_args()


def entropy(counts: Counter[str] | dict[str, int]) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    value = 0.0
    for count in counts.values():
        if count <= 0:
            continue
        p = count / total
        value -= p * math.log2(p)
    return value


def binary_mi(target_counts: Counter[str], present_counts: Counter[str], total: int) -> float:
    present_total = sum(present_counts.values())
    if total <= 0 or present_total <= 0 or present_total >= total:
        return 0.0

    absent_counts: Counter[str] = Counter()
    for target, count in target_counts.items():
        absent_counts[target] = count - present_counts.get(target, 0)

    h_target = entropy(target_counts)
    return h_target - (present_total / total) * entropy(present_counts) - ((total - present_total) / total) * entropy(absent_counts)


def categorical_mi(x_counts: Counter[str], y_counts: Counter[str], xy_counts: Counter[tuple[str, str]], total: int) -> float:
    if total <= 0:
        return 0.0
    value = 0.0
    for (x_value, y_value), count in xy_counts.items():
        if count <= 0:
            continue
        pxy = count / total
        px = x_counts[x_value] / total
        py = y_counts[y_value] / total
        if px and py:
            value += pxy * math.log2(pxy / (px * py))
    return value


def safe_ratio(num: float, den: float) -> float:
    return num / den if den else 0.0


def pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def fmt_float(value: float) -> str:
    return f"{value:.4f}"


def numeric_value(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float) and math.isfinite(float(value)):
        return float(value)
    return None


def iter_jsonl(path: Path, max_records: int) -> Iterable[tuple[int, dict[str, Any]]]:
    handle = sys.stdin if str(path) == "-" else path.open("r", encoding="utf-8")
    try:
        emitted = 0
        for line_number, line in enumerate(handle, 1):
            if max_records and emitted >= max_records:
                break
            line = line.strip()
            if not line:
                continue
            try:
                yield line_number, json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
            emitted += 1
    finally:
        if handle is not sys.stdin:
            handle.close()


def input_total(path: Path, max_records: int) -> int | None:
    if max_records:
        return max_records
    if str(path) == "-":
        return None
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def find_sensitive_input_paths(value: Any, prefix: str = "input") -> list[str]:
    paths: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{prefix}.{key}"
            if str(key).lower() in SENSITIVE_INPUT_KEYS:
                paths.append(child_path)
            paths.extend(find_sensitive_input_paths(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value[:100]):
            paths.extend(find_sensitive_input_paths(child, f"{prefix}[{index}]"))
    return paths


def feature_atoms(feature: dict[str, Any]) -> list[str]:
    feature_id = str(feature.get("id") or "<missing>")
    atoms: list[str] = []
    value = feature.get("value") or {}
    if not isinstance(value, dict):
        return atoms
    for key, raw in value.items():
        if key in SKIP_VALUE_KEYS:
            continue
        if key not in VALUE_ATOM_KEYS:
            continue
        values = raw if isinstance(raw, list) else [raw]
        for item in values[:20]:
            if isinstance(item, str | int | float | bool):
                item_s = str(item)
                if 0 < len(item_s) <= 80:
                    atoms.append(f"{feature_id}::value.{key}={item_s}")
    return atoms


def host_suffix(url: str) -> str:
    host = (urlparse(str(url or "")).hostname or "").lower()
    if not host:
        return ""
    parts = host.split(".")
    if len(parts) >= 2:
        return ".".join(parts[-2:])
    return host


def summarize_by_label(values: list[tuple[float, str, str]]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for value, label, _source in values:
        grouped[label].append(value)

    summary: dict[str, dict[str, float]] = {}
    for label, nums in grouped.items():
        nums = sorted(nums)
        if not nums:
            continue
        p90_index = min(len(nums) - 1, int(0.9 * (len(nums) - 1)))
        summary[label] = {
            "count": len(nums),
            "mean": statistics.fmean(nums),
            "median": statistics.median(nums),
            "p90": nums[p90_index],
        }
    return summary


def threshold_candidates(nums: list[float]) -> list[float]:
    unique = sorted(set(nums))
    if len(unique) <= 20:
        return unique
    candidates: list[float] = []
    for q in (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
        index = min(len(unique) - 1, max(0, int(q * (len(unique) - 1))))
        candidates.append(unique[index])
    return sorted(set(candidates))


def best_measurement_split(
    values: list[tuple[float, str, str]],
    label_counts: Counter[str],
    total: int,
) -> dict[str, Any] | None:
    if len(values) < 2:
        return None
    nums = [value for value, _label, _source in values]
    best: dict[str, Any] | None = None
    h_label = entropy(label_counts)
    for threshold in threshold_candidates(nums):
        ge_counts: Counter[str] = Counter()
        lt_counts: Counter[str] = Counter()
        for value, label, _source in values:
            if value >= threshold:
                ge_counts[label] += 1
            else:
                lt_counts[label] += 1
        mi = binary_mi(label_counts, ge_counts, total)
        nmi = safe_ratio(mi, h_label)
        ge_total = sum(ge_counts.values())
        lt_total = sum(lt_counts.values())
        ge_purity = max(ge_counts.values(), default=0) / ge_total if ge_total else 0.0
        lt_purity = max(lt_counts.values(), default=0) / lt_total if lt_total else 0.0
        if ge_purity >= lt_purity:
            branch_counts = ge_counts
            branch_support = ge_total
            operator = ">="
            branch_purity = ge_purity
        else:
            branch_counts = lt_counts
            branch_support = lt_total
            operator = "<"
            branch_purity = lt_purity
        branch_label = branch_counts.most_common(1)[0][0] if branch_counts else ""
        candidate = {
            "threshold": threshold,
            "operator": operator,
            "support": branch_support,
            "support_ratio": safe_ratio(branch_support, total),
            "label": branch_label,
            "purity": branch_purity,
            "mi": mi,
            "nmi": nmi,
        }
        if best is None or (candidate["nmi"], candidate["purity"], candidate["support"]) > (
            best["nmi"],
            best["purity"],
            best["support"],
        ):
            best = candidate
    return best


def risk_level(purity: float, nmi: float, source_nmi: float, args: argparse.Namespace) -> str:
    if purity >= args.high_purity or nmi >= args.high_nmi:
        return "high"
    if purity >= args.medium_purity or nmi >= args.medium_nmi or source_nmi >= args.medium_nmi:
        return "medium"
    return "low"


def has_label_variation(label_counts: Counter[str]) -> bool:
    return sum(1 for count in label_counts.values() if count > 0) >= 2


def risk_rank(risk: str) -> int:
    return {"high": 3, "medium": 2, "inconclusive": 1, "low": 0}.get(risk, 0)


def prevalence_text(counts: Counter[str], label_counts: Counter[str]) -> str:
    parts = []
    for label, label_total in label_counts.most_common():
        parts.append(f"{label}:{pct(safe_ratio(counts.get(label, 0), label_total))}")
    return ", ".join(parts)


def make_markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    if not rows:
        return "_No rows._\n"
    output = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        output.append("| " + " | ".join(str(item).replace("\n", " ") for item in row) + " |")
    return "\n".join(output) + "\n"


def build_feature_rows(
    item_label_counts: dict[str, Counter[str]],
    item_source_counts: dict[str, Counter[str]],
    label_counts: Counter[str],
    source_counts: Counter[str],
    supervision_types: dict[str, str],
    total: int,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    h_label = entropy(label_counts)
    h_source = entropy(source_counts)
    rows = []
    for item, present_label_counts in item_label_counts.items():
        support = sum(present_label_counts.values())
        if support < args.min_count:
            continue
        present_source_counts = item_source_counts.get(item, Counter())
        mi = binary_mi(label_counts, present_label_counts, total)
        source_mi = binary_mi(source_counts, present_source_counts, total)
        purity = max(present_label_counts.values(), default=0) / support
        top_label = present_label_counts.most_common(1)[0][0] if present_label_counts else ""
        nmi = safe_ratio(mi, h_label)
        source_nmi = safe_ratio(source_mi, h_source)
        rows.append(
            {
                "item": item,
                "support": support,
                "support_ratio": support / total,
                "top_label": top_label,
                "purity": purity,
                "label_entropy_present": entropy(present_label_counts),
                "label_nmi": nmi,
                "source_nmi": source_nmi,
                "risk": risk_level(purity, nmi, source_nmi, args) if has_label_variation(label_counts) else "inconclusive",
                "prevalence": prevalence_text(present_label_counts, label_counts),
                "supervision_type": supervision_types.get(item, ""),
            }
        )
    rows.sort(key=lambda row: (risk_rank(row["risk"]), row["label_nmi"], row["purity"], row["support"]), reverse=True)
    return rows


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    total = 0
    label_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    source_label_counts: Counter[tuple[str, str]] = Counter()
    input_sensitive_paths: Counter[str] = Counter()
    host_suffix_label_counts: dict[str, Counter[str]] = defaultdict(Counter)

    feature_label_counts: dict[str, Counter[str]] = defaultdict(Counter)
    feature_source_counts: dict[str, Counter[str]] = defaultdict(Counter)
    atom_label_counts: dict[str, Counter[str]] = defaultdict(Counter)
    atom_source_counts: dict[str, Counter[str]] = defaultdict(Counter)
    feature_pair_counts: Counter[tuple[str, str]] = Counter()
    feature_counts: Counter[str] = Counter()
    supervision_type_counts: Counter[str] = Counter()
    feature_supervision_types: dict[str, str] = {}
    measurement_values: dict[str, list[tuple[float, str, str]]] = defaultdict(list)

    total_hint = input_total(args.input, args.max_records)
    for _line_number, record in tqdm(
        iter_jsonl(args.input, args.max_records),
        total=total_hint,
        desc="analyze_features",
        unit="record",
        disable=args.no_progress,
        file=sys.stderr,
    ):
        total += 1
        label = str(record.get("label") or "<missing>")
        source = str(record.get("source") or "<missing>")
        label_counts[label] += 1
        source_counts[source] += 1
        source_label_counts[(source, label)] += 1

        page_input = record.get("input") or {}
        for path in find_sensitive_input_paths(page_input):
            input_sensitive_paths[path] += 1

        suffix = host_suffix(page_input.get("final_url") or page_input.get("url") or "")
        if suffix:
            host_suffix_label_counts[suffix][label] += 1

        features = record.get("features") or []
        feature_ids = sorted({str(feature.get("id") or "<missing>") for feature in features if isinstance(feature, dict)})
        for feature_id in feature_ids:
            feature_counts[feature_id] += 1
            feature_label_counts[feature_id][label] += 1
            feature_source_counts[feature_id][source] += 1

        for i, first in enumerate(feature_ids):
            for second in feature_ids[i + 1 :]:
                feature_pair_counts[(first, second)] += 1

        seen_atoms: set[str] = set()
        for feature in features:
            if not isinstance(feature, dict):
                continue
            feature_id = str(feature.get("id") or "<missing>")
            supervision = feature.get("supervision") or {}
            if isinstance(supervision, dict):
                supervision_type = str(supervision.get("type") or "")
                if supervision_type:
                    supervision_type_counts[supervision_type] += 1
                    feature_supervision_types.setdefault(feature_id, supervision_type)
            for atom in feature_atoms(feature):
                seen_atoms.add(atom)
        for atom in seen_atoms:
            atom_label_counts[atom][label] += 1
            atom_source_counts[atom][source] += 1

        measurements = record.get("measurements") or {}
        if isinstance(measurements, dict):
            for key, raw_value in measurements.items():
                value = numeric_value(raw_value)
                if value is not None:
                    measurement_values[str(key)].append((value, label, source))

    if total == 0:
        raise ValueError("No JSONL records were analyzed.")

    h_label = entropy(label_counts)
    h_source = entropy(source_counts)
    source_label_mi = categorical_mi(source_counts, label_counts, source_label_counts, total)
    source_label_nmi = safe_ratio(source_label_mi, h_label)

    feature_rows = build_feature_rows(
        feature_label_counts,
        feature_source_counts,
        label_counts,
        source_counts,
        feature_supervision_types,
        total,
        args,
    )
    atom_rows = build_feature_rows(
        atom_label_counts,
        atom_source_counts,
        label_counts,
        source_counts,
        {},
        total,
        args,
    )

    measurement_rows = []
    for key, values in measurement_values.items():
        if len(values) < args.min_count:
            continue
        best = best_measurement_split(values, label_counts, total)
        if not best:
            continue
        summary = summarize_by_label(values)
        measurement_rows.append(
            {
                "measurement": key,
                "count": len(values),
                "best": best,
                "risk": risk_level(best["purity"], best["nmi"], 0.0, args) if has_label_variation(label_counts) else "inconclusive",
                "summary": summary,
            }
        )
    measurement_rows.sort(
        key=lambda row: (risk_rank(row["risk"]), row["best"]["nmi"], row["best"]["purity"], row["best"]["support"]),
        reverse=True,
    )

    pair_rows = []
    for (first, second), pair_count in feature_pair_counts.items():
        if pair_count < args.min_count:
            continue
        first_count = feature_counts[first]
        second_count = feature_counts[second]
        jaccard = pair_count / (first_count + second_count - pair_count)
        conditional_a = pair_count / first_count if first_count else 0.0
        conditional_b = pair_count / second_count if second_count else 0.0
        if jaccard >= 0.6 or max(conditional_a, conditional_b) >= 0.9:
            pair_rows.append(
                {
                    "first": first,
                    "second": second,
                    "count": pair_count,
                    "jaccard": jaccard,
                    "p_second_given_first": conditional_a,
                    "p_first_given_second": conditional_b,
                }
            )
    pair_rows.sort(key=lambda row: (row["jaccard"], max(row["p_second_given_first"], row["p_first_given_second"]), row["count"]), reverse=True)

    source_rows = []
    for source, source_total in source_counts.most_common():
        counts = Counter({label: source_label_counts[(source, label)] for label in label_counts})
        top_label, top_count = counts.most_common(1)[0]
        source_rows.append(
            {
                "source": source,
                "count": source_total,
                "top_label": top_label,
                "purity": top_count / source_total if source_total else 0.0,
                "label_distribution": dict(counts),
            }
        )

    host_suffix_rows = build_feature_rows(
        {f"host_suffix={suffix}": counts for suffix, counts in host_suffix_label_counts.items()},
        defaultdict(Counter),
        label_counts,
        source_counts,
        {},
        total,
        args,
    )

    return {
        "total_records": total,
        "label_counts": dict(label_counts),
        "source_counts": dict(source_counts),
        "label_entropy": h_label,
        "source_entropy": h_source,
        "source_label_mi": source_label_mi,
        "source_label_nmi": source_label_nmi,
        "source_rows": source_rows,
        "input_sensitive_paths": dict(input_sensitive_paths),
        "supervision_type_counts": dict(supervision_type_counts),
        "feature_rows": feature_rows,
        "atom_rows": atom_rows,
        "measurement_rows": measurement_rows,
        "pair_rows": pair_rows,
        "host_suffix_rows": host_suffix_rows,
    }


def render_report(result: dict[str, Any], args: argparse.Namespace) -> str:
    lines: list[str] = []
    lines.append("# Feature Shortcut And Bias Audit\n")
    lines.append("## Summary\n")
    lines.append(f"- Records analyzed: {result['total_records']}")
    lines.append(f"- Label distribution: {result['label_counts']}")
    lines.append(f"- Source distribution: {result['source_counts']}")
    lines.append(f"- Label entropy: {fmt_float(result['label_entropy'])} bits")
    lines.append(f"- Source to label NMI: {fmt_float(result['source_label_nmi'])}")
    if result["label_entropy"] == 0:
        lines.append("- Risk metrics are inconclusive because only one label is present in this analysis slice. Use a balanced or shuffled slice to evaluate shortcut learning.")
    if result["source_label_nmi"] >= args.high_nmi:
        lines.append("- Risk: `source` is highly predictive of `label`. Do not include top-level `source` in model-visible input.")
    lines.append("")

    lines.append("## Leakage Checks\n")
    if result["input_sensitive_paths"]:
        rows = [[path, count] for path, count in Counter(result["input_sensitive_paths"]).most_common(args.top)]
        lines.append("Sensitive or label-like keys were found inside `input`:\n")
        lines.append(make_markdown_table(["Input path", "Records"], rows))
    else:
        lines.append("- No label/source/verdict-like keys were found inside `input`.\n")
    lines.append("- Top-level `label` and `source` are metadata. They are expected in this file but must not be shown to the LLM as input.\n")

    lines.append("## Source Label Bias\n")
    source_rows = [
        [row["source"], row["count"], row["top_label"], pct(row["purity"]), row["label_distribution"]]
        for row in result["source_rows"][: args.top]
    ]
    lines.append(make_markdown_table(["Source", "Count", "Top label", "Purity", "Labels"], source_rows))

    lines.append("## High-Risk Feature Shortcuts\n")
    high_feature_rows = [row for row in result["feature_rows"] if row["risk"] in {"high", "medium", "inconclusive"}][: args.top]
    lines.append(
        make_markdown_table(
            ["Risk", "Feature", "Count", "Top label", "Purity", "Label NMI", "Source NMI", "Prevalence by label", "Supervision"],
            [
                [
                    row["risk"],
                    row["item"],
                    row["support"],
                    row["top_label"],
                    pct(row["purity"]),
                    fmt_float(row["label_nmi"]),
                    fmt_float(row["source_nmi"]),
                    row["prevalence"],
                    row["supervision_type"],
                ]
                for row in high_feature_rows
            ],
        )
    )

    lines.append("## Measurement Threshold Shortcuts\n")
    measurement_rows = result["measurement_rows"][: args.top]
    lines.append(
        make_markdown_table(
            ["Risk", "Measurement", "Predicate", "Support", "Top label", "Purity", "Label NMI", "Per-label medians"],
            [
                [
                    row["risk"],
                    row["measurement"],
                    f"{row['measurement']} {row['best']['operator']} {row['best']['threshold']}",
                    f"{row['best']['support']} ({pct(row['best']['support_ratio'])})",
                    row["best"]["label"],
                    pct(row["best"]["purity"]),
                    fmt_float(row["best"]["nmi"]),
                    {
                        label: round(stats["median"], 4)
                        for label, stats in row["summary"].items()
                    },
                ]
                for row in measurement_rows
            ],
        )
    )

    lines.append("## Feature Value Shortcuts\n")
    atom_rows = [row for row in result["atom_rows"] if row["risk"] in {"high", "medium", "inconclusive"}][: args.top]
    lines.append(
        make_markdown_table(
            ["Risk", "Atom", "Count", "Top label", "Purity", "Label NMI", "Source NMI", "Prevalence by label"],
            [
                [
                    row["risk"],
                    row["item"],
                    row["support"],
                    row["top_label"],
                    pct(row["purity"]),
                    fmt_float(row["label_nmi"]),
                    fmt_float(row["source_nmi"]),
                    row["prevalence"],
                ]
                for row in atom_rows
            ],
        )
    )

    lines.append("## Host Suffix Shortcuts\n")
    host_rows = [row for row in result["host_suffix_rows"] if row["risk"] in {"high", "medium", "inconclusive"}][: args.top]
    lines.append(
        make_markdown_table(
            ["Risk", "Host suffix", "Count", "Top label", "Purity", "Label NMI", "Prevalence by label"],
            [
                [
                    row["risk"],
                    row["item"],
                    row["support"],
                    row["top_label"],
                    pct(row["purity"]),
                    fmt_float(row["label_nmi"]),
                    row["prevalence"],
                ]
                for row in host_rows
            ],
        )
    )

    lines.append("## Redundant Feature Pairs\n")
    lines.append(
        make_markdown_table(
            ["Feature A", "Feature B", "Count", "Jaccard", "P(B|A)", "P(A|B)"],
            [
                [
                    row["first"],
                    row["second"],
                    row["count"],
                    fmt_float(row["jaccard"]),
                    pct(row["p_second_given_first"]),
                    pct(row["p_first_given_second"]),
                ]
                for row in result["pair_rows"][: args.top]
            ],
        )
    )

    lines.append("## Supervision Types\n")
    supervision_rows = [[key, count] for key, count in Counter(result["supervision_type_counts"]).most_common()]
    lines.append(make_markdown_table(["Supervision type", "Feature instances"], supervision_rows))

    lines.append("## Interpretation Notes\n")
    lines.append("- High purity on a feature does not mean the feature is invalid. It means the model may be able to rely on that cue without learning the intended reasoning.")
    lines.append("- High source NMI is especially risky when source also predicts label. It usually indicates collection artifacts or source-specific extraction artifacts.")
    lines.append("- High-risk measurement thresholds should generally become raw calibration inputs, not model target evidence.")
    lines.append("- Host suffix rows are a domain memorization warning. Use domain/template holdouts to reduce this shortcut.\n")
    return "\n".join(lines)


def open_text_output(path: Path | None):
    if path is None or str(path) == "-":
        return sys.stdout
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.open("w", encoding="utf-8")


def main() -> int:
    args = parse_args()
    result = analyze(args)

    report = render_report(result, args)
    output = open_text_output(args.output)
    close_output = output is not sys.stdout
    try:
        output.write(report)
        if not report.endswith("\n"):
            output.write("\n")
    finally:
        if close_output:
            output.close()

    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
