#!/usr/bin/env python3
"""Build paper-oriented datasets directly from local MongoDB website documents.

These builders do not claim to exactly reconstruct the authors' original
collection pipelines. They create local, reproducible datasets from this
project's MongoDB crawls in the formats expected by the paper reproduction
workflows under `experimentation/`.

Targets:
- Hannousse and Yahiouche 2021: CSV using local classical page features
- Kapan and Gunal 2023: train/test CSVs with the paper's 25 feature schema
- EnLeM / UCI-style phishing websites: ARFF + CSV with UCI-like features
- HinPhish: HIN-inspired feature CSV built from structured page relations
"""

from __future__ import annotations

import argparse
import csv
import ipaddress
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable
from urllib.parse import urlparse

from dotenv import load_dotenv
from pymongo import MongoClient

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import extract_inputs_jsonl as stage_a
import fed_extractor as fed

EXPERIMENTATION_DIR = ROOT / "experimentation"

PAPER_DIRS = {
    "hannousse": EXPERIMENTATION_DIR / "hannousse_yahiouche_2021",
    "kapan": EXPERIMENTATION_DIR / "kapan_gunal_2023",
    "enlem": EXPERIMENTATION_DIR / "enlem_hlaing_2025_2026",
    "hinphish": EXPERIMENTATION_DIR / "guo_et_al_2021_hinphish",
}

URL_SHORTENERS = {
    "bit.ly",
    "buff.ly",
    "cutt.ly",
    "goo.gl",
    "is.gd",
    "lnkd.in",
    "ow.ly",
    "rebrand.ly",
    "rb.gy",
    "shorte.st",
    "shorturl.at",
    "t.co",
    "tiny.cc",
    "tinyurl.com",
    "v.gd",
}


def common_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--source",
        choices=["all", "phishing", "benign", "urlscan_live"],
        default="all",
        help="Mongo source to read.",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=ROOT / ".env",
        help="Path to .env containing MONGO_URI.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Mongo cursor batch size.",
    )
    parser.add_argument(
        "--limit-per-label",
        type=int,
        default=0,
        help="Maximum emitted rows per label. 0 means no cap.",
    )
    parser.add_argument(
        "--include-http-errors",
        action="store_true",
        help="Include HTTP 4xx/5xx pages. Default skips them.",
    )
    parser.add_argument(
        "--include-generic-error-pages",
        action="store_true",
        help="Include generic browser/server error pages. Default skips them.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=3407,
        help="Random seed for train/test splitting.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars.",
    )
    return parser


def runtime_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        batch_size=args.batch_size,
        include_http_errors=args.include_http_errors,
        include_generic_error_pages=args.include_generic_error_pages,
        no_progress=args.no_progress,
        max_items=100,
        max_visible_text_chars=4000,
        max_node_text_chars=300,
    )


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_parent(path)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    ensure_parent(path)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def load_client(env_file: Path) -> MongoClient:
    load_dotenv(env_file)
    mongo_uri = os.environ.get("MONGO_URI")
    if not mongo_uri:
        raise RuntimeError(f"MONGO_URI was not found in {env_file}")
    return MongoClient(mongo_uri, serverSelectionTimeoutMS=10000)


def iter_labeled_documents(
    client: MongoClient,
    args: argparse.Namespace,
) -> Iterable[tuple[dict[str, Any], str, str]]:
    fed_args = runtime_args(args)
    written = Counter()
    for label_name, db_name, collection_name in fed.selected_collections(args.source):
        source = f"{db_name}.{collection_name}"
        for doc in fed.iter_documents(client, db_name, collection_name, fed_args):
            label = fed.record_label_for_source(label_name, doc)
            if label not in {"phishing", "benign"}:
                continue
            if args.limit_per_label > 0 and written[label] >= args.limit_per_label:
                continue
            written[label] += 1
            yield fed.normalize_document(doc), label, source


def hostname(url: str) -> str:
    return fed.hostname(url or "")


def registrable_domain(url_or_host: str) -> str:
    extracted = fed.extract_tld(url_or_host)
    if extracted.domain and extracted.suffix:
        return f"{extracted.domain}.{extracted.suffix}".lower()
    if extracted.domain:
        return extracted.domain.lower()
    parsed = urlparse(url_or_host)
    host = parsed.netloc or parsed.path
    return host.lower()


def is_ip_host(url: str) -> bool:
    host = hostname(url)
    if not host:
        return False
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        return False


def count_subdomain_labels(url: str) -> int:
    host = hostname(url)
    if not host:
        return 0
    extracted = fed.extract_tld(host)
    full = host.split(".")
    suffix_parts = extracted.suffix.split(".") if extracted.suffix else []
    reserved = (1 if extracted.domain else 0) + len(suffix_parts)
    return max(0, len(full) - reserved)


def count_token(value: str, token: str) -> int:
    return (value or "").count(token)


def original_and_final_urls(normalized_doc: dict[str, Any]) -> tuple[str, str]:
    metadata = normalized_doc.get("metadata") or {}
    original_url = fed.collapse_ws(normalized_doc.get("url") or metadata.get("url"), 2000)
    final_url = fed.collapse_ws(metadata.get("final_url") or original_url, 2000)
    return original_url, final_url


def selector_for_doc(normalized_doc: dict[str, Any]):
    return fed.html_selector_from_text(normalized_doc.get("html") or "")


def status_code_of(normalized_doc: dict[str, Any]) -> int:
    metadata = normalized_doc.get("metadata") or {}
    status_code = metadata.get("status_code")
    return int(status_code) if isinstance(status_code, int) else 0


def has_redirect(normalized_doc: dict[str, Any], selector: Any, original_url: str, final_url: str) -> int:
    return fed.detect_redirection(normalized_doc, selector, normalized_doc.get("html") or "", original_url, final_url)


def classify_href(base_url: str, href: str) -> str:
    return fed.classify_link_target(base_url, href) or "null"


def count_tags(selector: Any, query: str) -> int:
    return fed.css_count(selector, query)


def lowercase_attr(tag: Any, name: str) -> str:
    return fed.lowercase_attr(tag, name)


def href_counts(selector: Any, base_url: str) -> tuple[int, int, int]:
    internal = 0
    external = 0
    total = 0
    for tag in fed.css_select(selector, "a"):
        href = fed.tag_attr(tag, "href")
        if href is None:
            continue
        total += 1
        kind = classify_href(base_url, href)
        if kind == "internal":
            internal += 1
        elif kind == "external":
            external += 1
    return internal, external, total


def same_domain_ratio(selector: Any, base_url: str, query: str, attr: str) -> tuple[int, int]:
    local = 0
    external = 0
    for tag in fed.css_select(selector, query):
        value = fed.tag_attr(tag, attr)
        if value is None:
            continue
        kind = classify_href(base_url, value)
        if kind == "internal":
            local += 1
        elif kind == "external":
            external += 1
    return local, external


def parse_known_date(value: Any) -> datetime | None:
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def iter_nested_events(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        if "eventAction" in value or "eventDate" in value:
            yield value
        for child in value.values():
            yield from iter_nested_events(child)
    elif isinstance(value, list):
        for item in value:
            yield from iter_nested_events(item)


def rdap_dates(normalized_doc: dict[str, Any]) -> tuple[datetime | None, datetime | None]:
    rdap = normalized_doc.get("rdap")
    creation = None
    expiry = None
    for event in iter_nested_events(rdap):
        action = str(event.get("eventAction") or "").lower()
        date = parse_known_date(event.get("eventDate"))
        if date is None:
            continue
        if creation is None and action in {"registration", "registered", "creation"}:
            creation = date
        if expiry is None and action in {"expiration", "expiry", "expiration date"}:
            expiry = date
    return creation, expiry


def fetched_time(normalized_doc: dict[str, Any]) -> datetime:
    for candidate in (
        normalized_doc.get("fetched_at"),
        normalized_doc.get("created_at"),
        (normalized_doc.get("metadata") or {}).get("fetched_at"),
    ):
        parsed = parse_known_date(candidate)
        if parsed is not None:
            return parsed
    return datetime.now(timezone.utc)


def infer_hannousse_rows(client: MongoClient, args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    label_counts = Counter()
    source_counts = Counter()
    feature_names: set[str] = set()
    for normalized_doc, label, source in iter_labeled_documents(client, args):
        page_url, _, features = fed.extract_features_from_document(normalized_doc)
        row = {"url": page_url}
        for name, value in features.items():
            if name == "whole_url_string":
                continue
            row[name] = value
            feature_names.add(name)
        row["status"] = "phishing" if label == "phishing" else "legitimate"
        rows.append(row)
        label_counts[label] += 1
        source_counts[source] += 1

    summary = {
        "rows": len(rows),
        "feature_count": len(feature_names),
        "label_counts": dict(label_counts),
        "source_counts": dict(source_counts),
    }
    return rows, summary


def kapan_feature_row(normalized_doc: dict[str, Any], label: str) -> dict[str, Any]:
    original_url, final_url = original_and_final_urls(normalized_doc)
    page_url = final_url or original_url
    selector = selector_for_doc(normalized_doc)
    original_domain = registrable_domain(original_url)
    final_domain = registrable_domain(final_url)
    similarity = round(SequenceMatcher(None, original_domain, final_domain).ratio(), 2)
    return {
        "domain_similarity": similarity,
        "url_length": len(page_url),
        "http_protocol": 1 if urlparse(page_url).scheme.lower() == "https" else 0,
        "num_dot": count_token(page_url, "."),
        "num_slash": count_token(page_url, "/"),
        "num_double_slash": count_token(page_url, "//"),
        "num_hyphen": count_token(page_url, "-"),
        "num_underscore": count_token(page_url, "_"),
        "num_equal": count_token(page_url, "="),
        "num_paranthesis": count_token(page_url, "(") + count_token(page_url, ")"),
        "num_curly_bracket": count_token(page_url, "{") + count_token(page_url, "}"),
        "num_square_bracket": count_token(page_url, "[") + count_token(page_url, "]"),
        "num_less_and_greater": count_token(page_url, "<") + count_token(page_url, ">"),
        "num_tilde": count_token(page_url, "~"),
        "num_asterisk": count_token(page_url, "*"),
        "num_plus": count_token(page_url, "+"),
        "url_inc_at": 1 if "@" in page_url else 0,
        "url_inc_ip": 1 if is_ip_host(page_url) else 0,
        "response_history": status_code_of(normalized_doc),
        "redirect": has_redirect(normalized_doc, selector, original_url, final_url),
        "num_a_href": count_tags(selector, "a"),
        "num_input": count_tags(selector, "input"),
        "num_button": count_tags(selector, "button"),
        "num_link_href": count_tags(selector, "link"),
        "num_iframe": count_tags(selector, "iframe"),
        "class": 1 if label == "phishing" else 0,
    }


def stratified_train_test(
    rows: list[dict[str, Any]],
    label_field: str,
    test_size: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row[label_field]].append(row)

    rng = random.Random(seed)
    train_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []

    for items in grouped.values():
        items = items[:]
        rng.shuffle(items)
        split_count = max(1, int(round(len(items) * test_size)))
        split_count = min(split_count, max(0, len(items) - 1))
        test_rows.extend(items[:split_count])
        train_rows.extend(items[split_count:])

    rng.shuffle(train_rows)
    rng.shuffle(test_rows)
    return train_rows, test_rows


def uci_like_row(normalized_doc: dict[str, Any], label: str) -> dict[str, int]:
    original_url, final_url = original_and_final_urls(normalized_doc)
    page_url = final_url or original_url
    selector = selector_for_doc(normalized_doc)
    parsed = urlparse(page_url)
    host = hostname(page_url)
    internal_links, external_links, href_total = href_counts(selector, page_url)
    script_local, script_external = same_domain_ratio(selector, page_url, "script[src]", "src")
    link_local, link_external = same_domain_ratio(selector, page_url, "link[href]", "href")
    media_local, media_external = same_domain_ratio(selector, page_url, "img[src],audio[src],video[src],source[src],embed[src]", "src")

    resource_total = script_local + script_external + link_local + link_external + media_local + media_external
    external_resource_ratio = (
        (script_external + link_external + media_external) / resource_total if resource_total else 0.0
    )
    external_anchor_ratio = external_links / href_total if href_total else 0.0

    creation_date, expiry_date = rdap_dates(normalized_doc)
    reference_time = fetched_time(normalized_doc)
    age_days = (reference_time - creation_date).days if creation_date else None
    remaining_days = (expiry_date - reference_time).days if expiry_date else None

    favicon_local, favicon_external = same_domain_ratio(selector, page_url, "link[rel*=icon][href]", "href")

    form_actions = []
    for form in fed.css_select(selector, "form"):
        action = fed.tag_attr(form, "action")
        if action is None:
            form_actions.append("null")
        else:
            form_actions.append(classify_href(page_url, action))

    def url_length_bucket(length: int) -> int:
        if length < 54:
            return 1
        if length <= 75:
            return 0
        return -1

    def subdomain_bucket(subdomain_count: int) -> int:
        if subdomain_count <= 1:
            return 1
        if subdomain_count == 2:
            return 0
        return -1

    def request_url_bucket(ratio: float) -> int:
        return 1 if ratio < 0.22 else 0 if ratio <= 0.61 else -1

    def anchor_bucket(ratio: float) -> int:
        return 1 if ratio < 0.31 else 0 if ratio <= 0.67 else -1

    def links_in_tags_bucket(ratio: float) -> int:
        return 1 if ratio < 0.17 else 0 if ratio <= 0.81 else -1

    def sfh_bucket(actions: list[str]) -> int:
        if not actions:
            return 0
        if any(action == "null" for action in actions):
            return -1
        if any(action == "external" for action in actions):
            return 0
        return 1

    return {
        "having_IP_Address": -1 if is_ip_host(page_url) else 1,
        "URL_Length": url_length_bucket(len(page_url)),
        "Shortining_Service": -1 if registrable_domain(page_url) in URL_SHORTENERS else 1,
        "having_At_Symbol": -1 if "@" in page_url else 1,
        "double_slash_redirecting": -1 if page_url.split("://", 1)[-1].count("//") > 0 else 1,
        "Prefix_Suffix": -1 if "-" in registrable_domain(page_url).split(".", 1)[0] else 1,
        "having_Sub_Domain": subdomain_bucket(count_subdomain_labels(page_url)),
        "SSLfinal_State": 1 if parsed.scheme.lower() == "https" else -1,
        "Domain_registeration_length": 1 if remaining_days is not None and remaining_days > 365 else -1 if remaining_days is not None else 0,
        "Favicon": 1 if favicon_external == 0 else -1,
        "port": -1 if parsed.port and parsed.port not in {80, 443} else 1,
        "HTTPS_token": -1 if "https" in host.replace("https://", "") and parsed.scheme.lower() != "https" else 1,
        "Request_URL": request_url_bucket(external_resource_ratio),
        "URL_of_Anchor": anchor_bucket(external_anchor_ratio),
        "Links_in_tags": links_in_tags_bucket(link_external / (link_local + link_external) if (link_local + link_external) else 0.0),
        "SFH": sfh_bucket(form_actions),
        "Submitting_to_email": -1 if "mailto:" in (normalized_doc.get("html") or "").lower() else 1,
        "Abnormal_URL": -1 if registrable_domain(original_url) != registrable_domain(final_url) else 1,
        "Redirect": 1 if has_redirect(normalized_doc, selector, original_url, final_url) else 0,
        "on_mouseover": -1 if "onmouseover" in (normalized_doc.get("html") or "").lower() else 1,
        "RightClick": -1 if re.search(r"contextmenu|event\.button\s*==\s*2|preventdefault\s*\(", (normalized_doc.get("html") or "").lower()) else 1,
        "popUpWidnow": -1 if fed.detect_alert_window(normalized_doc.get("html") or "") else 1,
        "Iframe": -1 if count_tags(selector, "iframe") > 0 else 1,
        "age_of_domain": 1 if age_days is not None and age_days >= 180 else -1 if age_days is not None else 0,
        "DNSRecord": 1 if normalized_doc.get("rdap") else -1,
        "web_traffic": 0,
        "Page_Rank": 0,
        "Google_Index": 0,
        "Links_pointing_to_page": 1 if internal_links > 2 else 0 if internal_links > 0 else -1,
        "Statistical_report": 0,
        "Result": -1 if label == "phishing" else 1,
    }


def classify_reference(base_url: str, reference: str) -> tuple[str, str]:
    lowered = (reference or "").strip().lower()
    if lowered in {"", "#", "/", "about:blank", "javascript:", "javascript:void(0)", "javascript:void(0);"}:
        return "null", "<null>"
    if lowered.startswith(("mailto:", "tel:", "javascript:")):
        return "null", "<null>"

    if reference.startswith("//"):
        target = reference
    else:
        try:
            parsed_reference = urlparse(reference)
        except ValueError:
            return "null", "<invalid_url>"
        if parsed_reference.scheme in {"http", "https"}:
            target = reference
        else:
            return "relative", "<relative>"

    try:
        target_domain = registrable_domain(target)
    except ValueError:
        return "null", "<invalid_url>"
    if classify_href(base_url, target) == "internal":
        return "local", target_domain
    return "foreign", target_domain


def relation_score(counts: Counter[str]) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    weighted = counts["foreign"] * 1.0 + counts["null"] * 1.0 + counts["relative"] * 0.25 + counts["local"] * -1.0
    return max(-1.0, min(1.0, weighted / total))


def hinphish_row(normalized_doc: dict[str, Any], label: str) -> dict[str, Any]:
    args = runtime_args(SimpleNamespace(
        batch_size=500,
        include_http_errors=False,
        include_generic_error_pages=False,
        no_progress=True,
    ))
    page_record = stage_a.build_input_record(normalized_doc, args, label, "mongodb")
    page = page_record["input"]
    base_url = page.get("final_url") or page.get("url") or ""

    domain_objects: dict[str, Counter[str]] = defaultdict(Counter)
    resource_objects: dict[str, Counter[str]] = defaultdict(Counter)

    for anchor in (page.get("anchors") or {}).get("items") or []:
        href = anchor.get("href") or ""
        relation, key = classify_reference(base_url, href)
        domain_objects[key][relation] += 1

    for form in (page.get("forms") or {}).get("items") or []:
        action = form.get("action") or ""
        relation, key = classify_reference(base_url, action)
        domain_objects[key][relation] += 1

    for key_name in ("favicon_hrefs", "script_src_sample", "stylesheet_href_sample", "image_src_sample"):
        for value in (page.get("resources") or {}).get(key_name) or []:
            relation, key = classify_reference(base_url, value)
            resource_objects[key][relation] += 1

    for iframe in (page.get("iframes") or {}).get("items") or []:
        src = iframe.get("src") or ""
        relation, key = classify_reference(base_url, src)
        resource_objects[key][relation] += 1

    all_objects: dict[tuple[str, str], Counter[str]] = {}
    for key, counts in domain_objects.items():
        all_objects[("domain", key)] = counts
    for key, counts in resource_objects.items():
        all_objects[("resource", key)] = counts

    initial_scores = {node: relation_score(counts) for node, counts in all_objects.items()}
    scores = initial_scores.copy()
    iterations = 0
    for iterations in range(1, 51):
        if not scores:
            break
        page_score = sum(scores.values()) / len(scores)
        new_scores: dict[tuple[str, str], float] = {}
        max_delta = 0.0
        for node, base_score in initial_scores.items():
            propagated = max(-1.0, min(1.0, 0.6 * base_score + 0.4 * page_score))
            new_scores[node] = propagated
            max_delta = max(max_delta, abs(propagated - scores[node]))
        scores = new_scores
        if max_delta < 1e-5:
            break

    domain_scores = [value for (kind, _), value in scores.items() if kind == "domain"]
    resource_scores = [value for (kind, _), value in scores.items() if kind == "resource"]
    all_scores = list(scores.values())

    def count_relation(objects: dict[str, Counter[str]], relation: str) -> int:
        return sum(counter[relation] for counter in objects.values())

    return {
        "id": page_record["id"],
        "label": 1 if label == "phishing" else 0,
        "label_name": label,
        "url": base_url,
        "score": round(sum(all_scores) / len(all_scores), 6) if all_scores else 0.0,
        "iters": iterations,
        "dom_local_link": count_relation(domain_objects, "local"),
        "dom_foreign_link": count_relation(domain_objects, "foreign"),
        "dom_null_link": count_relation(domain_objects, "null"),
        "dom_relative_link": count_relation(domain_objects, "relative"),
        "res_local_link": count_relation(resource_objects, "local"),
        "res_foreign_link": count_relation(resource_objects, "foreign"),
        "res_null_link": count_relation(resource_objects, "null"),
        "res_relative_link": count_relation(resource_objects, "relative"),
        "dom_mean": round(sum(domain_scores) / len(domain_scores), 6) if domain_scores else 0.0,
        "dom_var": round(_variance(domain_scores), 6),
        "mean": round(sum(all_scores) / len(all_scores), 6) if all_scores else 0.0,
        "var": round(_variance(all_scores), 6),
        "domain_object_count": len(domain_objects),
        "resource_object_count": len(resource_objects),
        "resource_mean": round(sum(resource_scores) / len(resource_scores), 6) if resource_scores else 0.0,
        "resource_var": round(_variance(resource_scores), 6),
    }


def _variance(values: list[float]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    return sum((value - mean) ** 2 for value in values) / len(values)


def write_arff(path: Path, relation: str, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"@relation {relation}\n\n")
        for name in fieldnames:
            handle.write(f"@attribute {name} numeric\n")
        handle.write("\n@data\n")
        for row in rows:
            handle.write(",".join(str(row[name]) for name in fieldnames))
            handle.write("\n")


def build_hannousse_dataset(args: argparse.Namespace) -> int:
    output = PAPER_DIRS["hannousse"] / "mongodb_dataset_B_like.csv"
    summary_path = PAPER_DIRS["hannousse"] / "mongodb_dataset_B_like_summary.json"
    with load_client(args.env_file) as client:
        rows, summary = infer_hannousse_rows(client, args)
    if not rows:
        raise RuntimeError("No Mongo documents produced a Hannousse-style dataset.")
    feature_names = sorted(key for key in rows[0].keys() if key not in {"url", "status"})
    write_csv(output, rows, ["url", *feature_names, "status"])
    write_json(summary_path, {**summary, "output": str(output)})
    return 0


def build_kapan_dataset(args: argparse.Namespace) -> int:
    full_output = PAPER_DIRS["kapan"] / "mongodb_full.csv"
    train_output = PAPER_DIRS["kapan"] / "mongodb_train.csv"
    test_output = PAPER_DIRS["kapan"] / "mongodb_test.csv"
    summary_path = PAPER_DIRS["kapan"] / "mongodb_summary.json"
    rows: list[dict[str, Any]] = []
    label_counts = Counter()
    source_counts = Counter()
    with load_client(args.env_file) as client:
        for normalized_doc, label, source in iter_labeled_documents(client, args):
            rows.append(kapan_feature_row(normalized_doc, label))
            label_counts[label] += 1
            source_counts[source] += 1
    if not rows:
        raise RuntimeError("No Mongo documents produced a Kapan-style dataset.")
    train_rows, test_rows = stratified_train_test(rows, "class", 0.30, args.seed)
    fieldnames = [
        "domain_similarity",
        "url_length",
        "http_protocol",
        "num_dot",
        "num_slash",
        "num_double_slash",
        "num_hyphen",
        "num_underscore",
        "num_equal",
        "num_paranthesis",
        "num_curly_bracket",
        "num_square_bracket",
        "num_less_and_greater",
        "num_tilde",
        "num_asterisk",
        "num_plus",
        "url_inc_at",
        "url_inc_ip",
        "response_history",
        "redirect",
        "num_a_href",
        "num_input",
        "num_button",
        "num_link_href",
        "num_iframe",
        "class",
    ]
    write_csv(full_output, rows, fieldnames)
    write_csv(train_output, train_rows, fieldnames)
    write_csv(test_output, test_rows, fieldnames)
    write_json(
        summary_path,
        {
            "rows": len(rows),
            "train_rows": len(train_rows),
            "test_rows": len(test_rows),
            "label_counts": dict(label_counts),
            "source_counts": dict(source_counts),
            "outputs": {
                "full": str(full_output),
                "train": str(train_output),
                "test": str(test_output),
            },
        },
    )
    return 0


def build_enlem_dataset(args: argparse.Namespace) -> int:
    csv_output = PAPER_DIRS["enlem"] / "mongodb_phishing_websites.csv"
    arff_output = PAPER_DIRS["enlem"] / "mongodb_phishing_websites.arff"
    summary_path = PAPER_DIRS["enlem"] / "mongodb_summary.json"
    rows: list[dict[str, Any]] = []
    label_counts = Counter()
    source_counts = Counter()
    with load_client(args.env_file) as client:
        for normalized_doc, label, source in iter_labeled_documents(client, args):
            rows.append(uci_like_row(normalized_doc, label))
            label_counts[label] += 1
            source_counts[source] += 1
    if not rows:
        raise RuntimeError("No Mongo documents produced an EnLeM/UCI-style dataset.")
    fieldnames = [
        "having_IP_Address",
        "URL_Length",
        "Shortining_Service",
        "having_At_Symbol",
        "double_slash_redirecting",
        "Prefix_Suffix",
        "having_Sub_Domain",
        "SSLfinal_State",
        "Domain_registeration_length",
        "Favicon",
        "port",
        "HTTPS_token",
        "Request_URL",
        "URL_of_Anchor",
        "Links_in_tags",
        "SFH",
        "Submitting_to_email",
        "Abnormal_URL",
        "Redirect",
        "on_mouseover",
        "RightClick",
        "popUpWidnow",
        "Iframe",
        "age_of_domain",
        "DNSRecord",
        "web_traffic",
        "Page_Rank",
        "Google_Index",
        "Links_pointing_to_page",
        "Statistical_report",
        "Result",
    ]
    write_csv(csv_output, rows, fieldnames)
    write_arff(arff_output, "mongodb_phishing_websites", fieldnames, rows)
    write_json(
        summary_path,
        {
            "rows": len(rows),
            "label_counts": dict(label_counts),
            "source_counts": dict(source_counts),
            "outputs": {"csv": str(csv_output), "arff": str(arff_output)},
        },
    )
    return 0


def build_hinphish_dataset(args: argparse.Namespace) -> int:
    csv_output = PAPER_DIRS["hinphish"] / "mongodb_hinphish_features.csv"
    summary_path = PAPER_DIRS["hinphish"] / "mongodb_summary.json"
    rows: list[dict[str, Any]] = []
    label_counts = Counter()
    source_counts = Counter()
    with load_client(args.env_file) as client:
        for normalized_doc, label, source in iter_labeled_documents(client, args):
            rows.append(hinphish_row(normalized_doc, label))
            label_counts[label] += 1
            source_counts[source] += 1
    if not rows:
        raise RuntimeError("No Mongo documents produced a HinPhish-style dataset.")
    fieldnames = [
        "id",
        "label",
        "label_name",
        "url",
        "score",
        "iters",
        "dom_local_link",
        "dom_foreign_link",
        "dom_null_link",
        "dom_relative_link",
        "res_local_link",
        "res_foreign_link",
        "res_null_link",
        "res_relative_link",
        "dom_mean",
        "dom_var",
        "mean",
        "var",
        "domain_object_count",
        "resource_object_count",
        "resource_mean",
        "resource_var",
    ]
    write_csv(csv_output, rows, fieldnames)
    write_json(
        summary_path,
        {
            "rows": len(rows),
            "label_counts": dict(label_counts),
            "source_counts": dict(source_counts),
            "output": str(csv_output),
        },
    )
    return 0


def main_hannousse() -> int:
    parser = common_parser("Build a Hannousse-style CSV from MongoDB website documents.")
    return build_hannousse_dataset(parser.parse_args())


def main_kapan() -> int:
    parser = common_parser("Build Kapan-style train/test CSVs from MongoDB website documents.")
    return build_kapan_dataset(parser.parse_args())


def main_enlem() -> int:
    parser = common_parser("Build an EnLeM/UCI-style ARFF+CSV dataset from MongoDB website documents.")
    return build_enlem_dataset(parser.parse_args())


def main_hinphish() -> int:
    parser = common_parser("Build a HinPhish-style feature CSV from MongoDB website documents.")
    return build_hinphish_dataset(parser.parse_args())


def main() -> int:
    parser = common_parser("Build one or more paper-oriented datasets from MongoDB website documents.")
    parser.add_argument(
        "--target",
        choices=["all", "hannousse", "kapan", "enlem", "hinphish"],
        default="all",
        help="Dataset target to build.",
    )
    args = parser.parse_args()
    if args.target in {"all", "hannousse"}:
        build_hannousse_dataset(args)
    if args.target in {"all", "kapan"}:
        build_kapan_dataset(args)
    if args.target in {"all", "enlem"}:
        build_enlem_dataset(args)
    if args.target in {"all", "hinphish"}:
        build_hinphish_dataset(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
