#!/usr/bin/env python3
"""Extract model input records from MongoDB as JSONL.

This script intentionally does not extract phishing features, verdicts, or
natural-language evidence statements. It only converts fetched website
documents into compact structured observations suitable for a later feature
target-generation step.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from dotenv import load_dotenv
from parsel import Selector, SelectorList
from pymongo import MongoClient
from tqdm import tqdm


DEFAULT_COLLECTIONS = {
    "phishing": ("phishing_db", "website_content"),
    "benign": ("tranco", "websites"),
    "urlscan_live": ("urlscan", "live"),
}

GENERIC_ERROR_TITLES = {
    "403 forbidden",
    "404 not found",
    "access denied",
    "privacy error",
    "account disabled by server administrator",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write JSONL records containing only the LLM input object."
    )
    parser.add_argument(
        "--source",
        choices=["all", "phishing", "benign", "urlscan_live"],
        default="all",
        help="Mongo source to read. This is not written to the JSONL records.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSONL file. Defaults to stdout.",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path(".env"),
        help="Path to .env containing MONGO_URI.",
    )
    parser.add_argument(
        "--limit-per-collection",
        type=int,
        default=0,
        help="Maximum records per output label. 0 means no limit.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Mongo cursor batch size.",
    )
    parser.add_argument(
        "--max-visible-text-chars",
        type=int,
        default=4000,
        help="Maximum normalized visible text characters per record.",
    )
    parser.add_argument(
        "--max-node-text-chars",
        type=int,
        default=300,
        help="Maximum normalized text characters for one form/link/iframe item.",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=100,
        help="Maximum meta/form/anchor/iframe/resource items kept per record.",
    )
    parser.add_argument(
        "--include-http-errors",
        action="store_true",
        help="Include HTTP 4xx/5xx pages. By default they are skipped.",
    )
    parser.add_argument(
        "--include-generic-error-pages",
        action="store_true",
        help="Include generic browser/server error pages. By default they are skipped.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars.",
    )
    return parser.parse_args()


def collapse_ws(value: Any, max_chars: int | None = None) -> str:
    if value is None:
        text = ""
    else:
        text = str(value)
    text = re.sub(r"\s+", " ", text).strip()
    if max_chars is not None and len(text) > max_chars:
        return text[:max_chars].rstrip()
    return text


def optional_text(value: Any, max_chars: int | None = None) -> str | None:
    text = collapse_ws(value, max_chars)
    return text or None


def stable_record_id(doc: dict[str, Any]) -> str:
    raw_id = doc.get("_id")
    if raw_id is not None:
        return str(raw_id)

    seed = "|".join(
        [
            str(doc.get("url") or ""),
            str((doc.get("metadata") or {}).get("final_url") or ""),
            str(doc.get("title") or ""),
        ]
    )
    return hashlib.sha256(seed.encode("utf-8", errors="ignore")).hexdigest()[:24]


def first_non_empty(*values: Any) -> Any:
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        if isinstance(value, (dict, list, tuple, set)) and not value:
            continue
        return value
    return None


def tag_attr(tag: Any, name: str) -> str | None:
    try:
        value = tag.root.get(name)
    except (AttributeError, TypeError, ValueError):
        value = None
    if isinstance(value, list):
        value = " ".join(str(item) for item in value)
    return optional_text(value, 500)


def html_selector_from_text(html: str) -> Selector:
    selector = Selector(text=html, type="html")
    if selector.type != "html":
        selector = Selector(
            text=f"<!doctype html><html><body>{html}</body></html>",
            type="html",
        )
    return selector


def css_select(selector: Any, query: str) -> SelectorList:
    try:
        return selector.css(query)
    except (AttributeError, TypeError, ValueError):
        return SelectorList([])


def css_first_text(selector: Any, query: str) -> str | None:
    matches = css_select(selector, query)
    if not matches:
        return None
    return matches[0].get()


def selector_text(selector: Selector, max_chars: int | None = None) -> str | None:
    try:
        text_parts = selector.xpath(
            ".//text()[not(ancestor::script) and not(ancestor::style) "
            "and not(ancestor::noscript) and not(ancestor::svg) "
            "and not(ancestor::template)]"
        ).getall()
    except (AttributeError, TypeError, ValueError):
        text_parts = []
    return optional_text(" ".join(text_parts), max_chars)


def visible_text_from_selector(selector: Selector, max_chars: int) -> str:
    body_text = selector_text(css_select(selector, "body"), max_chars)
    if body_text:
        return body_text
    return selector_text(selector, max_chars) or ""


def extract_meta(selector: Selector, max_items: int) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    for tag in css_select(selector, "meta")[:max_items]:
        item: dict[str, str] = {}
        for attr in ("name", "property", "http-equiv", "charset", "content"):
            value = tag_attr(tag, attr)
            if value:
                item["http_equiv" if attr == "http-equiv" else attr] = value
        if item:
            items.append(item)
    return items


def label_text_for_input(form: Any, input_tag: Any) -> str | None:
    input_id = tag_attr(input_tag, "id")
    if input_id:
        label = form.xpath(".//label[@for=$input_id]", input_id=input_id)
        if label:
            return selector_text(label, 160)

    parent_label = input_tag.xpath("ancestor::label[1]")
    if parent_label:
        return selector_text(parent_label, 160)

    return None


def extract_inputs(form: Any, max_items: int) -> list[dict[str, str]]:
    inputs: list[dict[str, str]] = []
    for input_tag in css_select(form, "input, textarea, select")[:max_items]:
        item: dict[str, str] = {"tag": input_tag.root.tag}
        for attr in (
            "type",
            "name",
            "id",
            "placeholder",
            "autocomplete",
            "aria-label",
        ):
            value = tag_attr(input_tag, attr)
            if value:
                item["aria_label" if attr == "aria-label" else attr] = value

        label = label_text_for_input(form, input_tag)
        if label:
            item["label"] = label

        # Do not copy arbitrary input values. Hidden/text field values can contain
        # victim identifiers or crawler artifacts and are not needed as input text.
        if input_tag.root.tag == "select":
            options = [
                selector_text(option, 80) or ""
                for option in css_select(input_tag, "option")[:10]
            ]
            options = [option for option in options if option]
            if options:
                item["options_sample"] = " | ".join(options)

        inputs.append(item)
    return inputs


def extract_buttons(form: Any, max_items: int) -> list[str]:
    buttons: list[str] = []
    for button in css_select(form, "button, input")[:max_items]:
        tag_name = button.root.tag
        input_type = (tag_attr(button, "type") or "").lower()
        if tag_name == "input" and input_type not in {"submit", "button", "reset"}:
            continue
        text = selector_text(button, 120)
        if not text:
            text = optional_text(tag_attr(button, "value"), 120)
        if text:
            buttons.append(text)
    return buttons


def extract_forms(
    selector: Selector, max_items: int, max_node_text_chars: int
) -> dict[str, Any]:
    forms: list[dict[str, Any]] = []
    all_forms = css_select(selector, "form")
    for form in all_forms[:max_items]:
        item: dict[str, Any] = {}
        for attr in ("method", "action", "name", "id"):
            value = tag_attr(form, attr)
            if value:
                item[attr] = value

        text = selector_text(form, max_node_text_chars)
        if text:
            item["text"] = text

        inputs = extract_inputs(form, max_items)
        if inputs:
            item["inputs"] = inputs

        buttons = extract_buttons(form, max_items)
        if buttons:
            item["buttons"] = buttons

        forms.append(item)

    return {
        "total_observed": len(all_forms),
        "items": forms,
    }


def extract_anchors(
    selector: Selector, max_items: int, max_node_text_chars: int
) -> dict[str, Any]:
    anchors: list[dict[str, str]] = []
    all_anchors = css_select(selector, "a")
    for anchor in all_anchors[:max_items]:
        item: dict[str, str] = {}
        href = tag_attr(anchor, "href")
        text = selector_text(anchor, max_node_text_chars)
        title = tag_attr(anchor, "title")
        aria_label = tag_attr(anchor, "aria-label")
        if text:
            item["text"] = text
        if href is not None:
            item["href"] = href
        if title:
            item["title"] = title
        if aria_label:
            item["aria_label"] = aria_label
        if item:
            anchors.append(item)

    return {
        "total_observed": len(all_anchors),
        "items": anchors,
    }


def extract_iframes(selector: Selector, max_items: int) -> dict[str, Any]:
    iframes: list[dict[str, str]] = []
    all_iframes = css_select(selector, "iframe")
    for iframe in all_iframes[:max_items]:
        item: dict[str, str] = {}
        for attr in ("src", "name", "id", "title", "width", "height", "style"):
            value = tag_attr(iframe, attr)
            if value:
                item[attr] = value
        if item:
            iframes.append(item)

    return {
        "total_observed": len(all_iframes),
        "items": iframes,
    }


def extract_resources(selector: Selector, max_items: int) -> dict[str, Any]:
    favicon_hrefs: list[str] = []
    stylesheet_hrefs: list[str] = []
    script_srcs: list[str] = []
    image_srcs: list[str] = []

    for link in css_select(selector, "link"):
        rel = (tag_attr(link, "rel") or "").lower()
        href = tag_attr(link, "href")
        if not href:
            continue
        if "icon" in rel:
            favicon_hrefs.append(href)
        elif "stylesheet" in rel:
            stylesheet_hrefs.append(href)

    for script in css_select(selector, "script"):
        src = tag_attr(script, "src")
        if src:
            script_srcs.append(src)

    for image in css_select(selector, "img"):
        src = tag_attr(image, "src")
        if src:
            image_srcs.append(src)

    return {
        "favicon_hrefs": favicon_hrefs[:max_items],
        "stylesheet_href_sample": stylesheet_hrefs[:max_items],
        "script_src_sample": script_srcs[:max_items],
        "image_src_sample": image_srcs[:max_items],
    }


def extract_redirects(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    redirects: list[dict[str, Any]] = []
    for redirect in metadata.get("redirect_history") or []:
        if not isinstance(redirect, dict):
            continue
        item: dict[str, Any] = {}
        status_code = redirect.get("status_code")
        url = optional_text(redirect.get("url"), 2000)
        if status_code is not None:
            item["status_code"] = status_code
        if url:
            item["url"] = url
        if item:
            redirects.append(item)
    return redirects


def title_from_doc_or_html(doc: dict[str, Any], selector: Selector) -> str | None:
    title = optional_text(doc.get("title"), 300)
    if title:
        return title
    return optional_text(css_first_text(selector, "title::text"), 300)


def decode_html_blob(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (bytes, bytearray)):
        try:
            return gzip.decompress(value).decode("utf-8", errors="replace")
        except (OSError, EOFError, TypeError, ValueError):
            return value.decode("utf-8", errors="replace")
    return ""


def extract_redirects_from_urlscan(doc: dict[str, Any]) -> list[dict[str, Any]]:
    requests = (
        ((doc.get("urlscanresults") or {}).get("data") or {}).get("requests") or []
    )
    redirects: list[dict[str, Any]] = []
    seen: set[tuple[Any, Any]] = set()
    for request_group in requests:
        if not isinstance(request_group, dict):
            continue
        for request in request_group.get("requests") or []:
            if not isinstance(request, dict):
                continue
            redirect_response = request.get("redirectResponse") or {}
            status_code = redirect_response.get("status")
            url = optional_text(redirect_response.get("url"), 2000)
            item: dict[str, Any] = {}
            if status_code is not None:
                item["status_code"] = status_code
            if url:
                item["url"] = url
            if not item:
                continue
            key = (item.get("status_code"), item.get("url"))
            if key in seen:
                continue
            seen.add(key)
            redirects.append(item)
    return redirects


def selected_urlscan_scan(doc: dict[str, Any]) -> dict[str, Any] | None:
    scans = doc.get("scans") or {}
    t7 = scans.get("T7")
    if isinstance(t7, dict):
        return t7
    t0 = scans.get("T0")
    if isinstance(t0, dict):
        return t0
    return None


def urlscan_label(doc: dict[str, Any]) -> str | None:
    scan = selected_urlscan_scan(doc)
    if not isinstance(scan, dict):
        return None

    virustotal = scan.get("virustotal")
    if isinstance(virustotal, dict) and virustotal:
        malicious_engine_count = virustotal.get("malicious_engine_count")
        if isinstance(malicious_engine_count, int):
            return "phishing" if malicious_engine_count > 5 else "benign"

    gsb = scan.get("google_safe_browsing")
    if isinstance(gsb, dict) and gsb:
        matched = gsb.get("matched")
        status = collapse_ws(gsb.get("status"), 100).lower()
        if matched is True or status == "malicious":
            return "phishing"
    return None


def normalize_document(doc: dict[str, Any]) -> dict[str, Any]:
    if isinstance(doc.get("html"), str):
        return doc

    task = first_non_empty(doc.get("task"), (doc.get("urlscanresults") or {}).get("task"))
    page = first_non_empty(doc.get("page"), (doc.get("urlscanresults") or {}).get("page"))
    task = task if isinstance(task, dict) else {}
    page = page if isinstance(page, dict) else {}
    dom = doc.get("dom") if isinstance(doc.get("dom"), dict) else {}
    metadata = {
        "url": optional_text(task.get("url"), 2000),
        "final_url": optional_text(
            first_non_empty(page.get("url"), task.get("url")),
            2000,
        ),
        "redirect_history": extract_redirects_from_urlscan(doc),
        "status_code": page.get("status"),
    }
    normalized = {
        "_id": doc.get("_id"),
        "url": optional_text(task.get("url"), 2000),
        "title": optional_text(page.get("title"), 300),
        "html": decode_html_blob(dom.get("data")),
        "error": None,
        "metadata": metadata,
    }
    return normalized


def build_input_record(
    doc: dict[str, Any],
    args: argparse.Namespace,
    label: str,
    source: str,
) -> dict[str, Any]:
    normalized_doc = normalize_document(doc)
    html = normalized_doc.get("html") or ""
    metadata = normalized_doc.get("metadata") or {}
    selector = html_selector_from_text(html)

    page_input: dict[str, Any] = {
        "url": collapse_ws(normalized_doc.get("url") or metadata.get("url"), 2000),
        "final_url": collapse_ws(
            metadata.get("final_url") or normalized_doc.get("url"),
            2000,
        ),
        "redirects": extract_redirects(metadata),
        "title": title_from_doc_or_html(normalized_doc, selector),
        "meta": extract_meta(selector, args.max_items),
        "visible_text": visible_text_from_selector(selector, args.max_visible_text_chars),
        "forms": extract_forms(selector, args.max_items, args.max_node_text_chars),
        "anchors": extract_anchors(selector, args.max_items, args.max_node_text_chars),
        "iframes": extract_iframes(selector, args.max_items),
        "resources": extract_resources(selector, args.max_items),
    }

    # Remove empty optional values while preserving empty containers used by the
    # downstream input schema.
    return {
        "id": stable_record_id(normalized_doc),
        "label": label,
        "source": source,
        "input": {key: value for key, value in page_input.items() if value is not None},
    }


def should_skip_doc(doc: dict[str, Any], args: argparse.Namespace) -> bool:
    normalized_doc = normalize_document(doc)
    html = normalized_doc.get("html")
    if not isinstance(html, str) or not html.strip():
        return True

    if normalized_doc.get("error") is not None:
        return True

    metadata = normalized_doc.get("metadata") or {}
    if metadata.get("error") is not None:
        return True

    status_code = metadata.get("status_code")
    if (
        not args.include_http_errors
        and isinstance(status_code, int)
        and status_code >= 400
    ):
        return True

    if not args.include_generic_error_pages:
        title = collapse_ws(normalized_doc.get("title"), 200).lower()
        if title in GENERIC_ERROR_TITLES:
            return True

    return False


def selected_collections(source: str) -> list[tuple[str, str, str]]:
    if source == "all":
        return [
            (label, db_name, collection_name)
            for label, (db_name, collection_name) in DEFAULT_COLLECTIONS.items()
        ]
    db_name, collection_name = DEFAULT_COLLECTIONS[source]
    return [(source, db_name, collection_name)]


def selected_output_labels(source: str) -> set[str]:
    if source == "all":
        return {"phishing", "benign"}
    if source == "urlscan_live":
        return {"phishing", "benign"}
    return {source}


def collection_possible_labels(label_name: str) -> set[str]:
    if label_name == "urlscan_live":
        return {"phishing", "benign"}
    return {label_name}


def mongo_query() -> dict[str, Any]:
    return {
        "$or": [
            {"html": {"$exists": True, "$type": "string", "$ne": ""}},
            {"dom.data": {"$exists": True, "$ne": None}},
        ]
    }


def mongo_projection() -> dict[str, int]:
    return {
        "_id": 1,
        "url": 1,
        "title": 1,
        "html": 1,
        "error": 1,
        "metadata": 1,
        "task": 1,
        "page": 1,
        "scans.T0": 1,
        "scans.T7": 1,
        "dom.data": 1,
        "urlscanresults.task": 1,
        "urlscanresults.page": 1,
        "urlscanresults.data.requests": 1,
    }


def standard_source_count_query(args: argparse.Namespace) -> dict[str, Any]:
    conditions: list[dict[str, Any]] = [
        {"html": {"$exists": True, "$type": "string", "$ne": ""}},
        {"$or": [{"error": {"$exists": False}}, {"error": None}]},
        {"$or": [{"metadata.error": {"$exists": False}}, {"metadata.error": None}]},
    ]
    if not args.include_http_errors:
        conditions.append(
            {
                "$or": [
                    {"metadata.status_code": {"$exists": False}},
                    {"metadata.status_code": {"$lt": 400}},
                ]
            }
        )
    return {"$and": conditions}


def record_label_for_source(label_name: str, doc: dict[str, Any]) -> str | None:
    if label_name != "urlscan_live":
        return label_name
    return urlscan_label(doc)


def count_available_documents(
    client: MongoClient,
    db_name: str,
    collection_name: str,
    args: argparse.Namespace,
    label_name: str,
) -> Counter[str]:
    if label_name == "urlscan_live":
        return count_urlscan_documents(client, db_name, collection_name, args)

    collection = client[db_name][collection_name]
    return Counter({label_name: collection.count_documents(standard_source_count_query(args))})


def iter_documents(
    client: MongoClient, db_name: str, collection_name: str, args: argparse.Namespace
) -> Iterable[dict[str, Any]]:
    collection = client[db_name][collection_name]
    cursor = collection.find(
        mongo_query(),
        mongo_projection(),
        batch_size=args.batch_size,
    )
    progress = tqdm(
        desc=f"{db_name}.{collection_name}",
        unit="doc",
        disable=args.no_progress,
        file=sys.stderr,
    )
    try:
        for doc in cursor:
            progress.update(1)
            if should_skip_doc(doc, args):
                continue
            yield doc
    finally:
        progress.close()


def random_source_quotas(
    available_by_label: dict[str, Counter[str]],
    target_labels: set[str],
    limit_per_label: int,
) -> Counter[tuple[str, str]]:
    quotas: Counter[tuple[str, str]] = Counter()
    if limit_per_label <= 0:
        for label in target_labels:
            for source, count in available_by_label.get(label, {}).items():
                quotas[(source, label)] = count
        return quotas

    rng = random.Random()
    for label in target_labels:
        remaining = Counter(available_by_label.get(label, {}))
        draws = min(limit_per_label, sum(remaining.values()))
        for _ in range(draws):
            sources = list(remaining.keys())
            weights = [remaining[source] for source in sources]
            selected_source = rng.choices(sources, weights=weights, k=1)[0]
            quotas[(selected_source, label)] += 1
            remaining[selected_source] -= 1
            if remaining[selected_source] <= 0:
                del remaining[selected_source]
    return quotas


def urlscan_base_match(args: argparse.Namespace) -> list[dict[str, Any]]:
    del args
    return []


def urlscan_label_expression() -> dict[str, Any]:
    return {
        "$let": {
            "vars": {"selected_scan": {"$ifNull": ["$scans.T7", "$scans.T0"]}},
            "in": {
                "$cond": [
                    {
                        "$in": [
                            {"$type": "$$selected_scan.virustotal.malicious_engine_count"},
                            ["int", "long", "double", "decimal"],
                        ]
                    },
                    {
                        "$cond": [
                            {"$gt": ["$$selected_scan.virustotal.malicious_engine_count", 5]},
                            "phishing",
                            "benign",
                        ]
                    },
                    {
                        "$cond": [
                            {
                                "$or": [
                                    {"$eq": ["$$selected_scan.google_safe_browsing.matched", True]},
                                    {
                                        "$eq": [
                                            {
                                                "$toLower": {
                                                    "$ifNull": [
                                                        "$$selected_scan.google_safe_browsing.status",
                                                        "",
                                                    ]
                                                }
                                            },
                                            "malicious",
                                        ]
                                    },
                                ]
                            },
                            "phishing",
                            None,
                        ]
                    },
                ]
            },
        }
    }


def count_urlscan_documents(
    client: MongoClient,
    db_name: str,
    collection_name: str,
    args: argparse.Namespace,
) -> Counter[str]:
    collection = client[db_name][collection_name]
    pipeline: list[dict[str, Any]] = []
    for match in urlscan_base_match(args):
        pipeline.append({"$match": match})
    pipeline.extend(
        [
            {"$addFields": {"derived_label": urlscan_label_expression()}},
            {"$match": {"derived_label": {"$in": ["phishing", "benign"]}}},
            {"$group": {"_id": "$derived_label", "count": {"$sum": 1}}},
        ]
    )
    counts: Counter[str] = Counter()
    for row in collection.aggregate(pipeline, allowDiskUse=True):
        counts[str(row["_id"])] = int(row["count"])
    return counts


def sample_urlscan_documents(
    client: MongoClient,
    db_name: str,
    collection_name: str,
    args: argparse.Namespace,
    label: str,
    sample_size: int,
) -> list[dict[str, Any]]:
    if sample_size <= 0:
        return []
    collection = client[db_name][collection_name]
    pipeline: list[dict[str, Any]] = []
    for match in urlscan_base_match(args):
        pipeline.append({"$match": match})
    pipeline.extend(
        [
            {"$addFields": {"derived_label": urlscan_label_expression()}},
            {"$match": {"derived_label": label}},
            {"$sample": {"size": max(sample_size * 5, sample_size)}},
            {"$project": mongo_projection()},
        ]
    )
    docs = [
        doc
        for doc in collection.aggregate(pipeline, allowDiskUse=True)
        if not should_skip_doc(doc, args)
    ]
    return docs[:sample_size]


def open_output(path: Path | None):
    if path is None:
        return sys.stdout

    path.parent.mkdir(parents=True, exist_ok=True)
    return path.open("w", encoding="utf-8")


def main() -> int:
    args = parse_args()
    load_dotenv(args.env_file)

    import os

    mongo_uri = os.environ.get("MONGO_URI")
    if not mongo_uri:
        print(f"MONGO_URI was not found in {args.env_file}", file=sys.stderr)
        return 2

    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=10000)
    output = open_output(args.output)
    close_output = output is not sys.stdout

    try:
        written = 0
        collections = selected_collections(args.source)
        target_labels = selected_output_labels(args.source)
        if args.limit_per_collection > 0:
            available_by_label: dict[str, Counter[str]] = {
                label: Counter() for label in target_labels
            }
            for label_name, db_name, collection_name in collections:
                source = f"{db_name}.{collection_name}"
                for record_label, count in count_available_documents(
                    client,
                    db_name,
                    collection_name,
                    args,
                    label_name,
                ).items():
                    if record_label in target_labels:
                        available_by_label[record_label][source] = count

            quotas = random_source_quotas(
                available_by_label,
                target_labels,
                args.limit_per_collection,
            )
        else:
            quotas = Counter()
            for label_name, db_name, collection_name in collections:
                source = f"{db_name}.{collection_name}"
                for label in collection_possible_labels(label_name):
                    quotas[(source, label)] = float("inf")

        for label_name, db_name, collection_name in collections:
            source = f"{db_name}.{collection_name}"
            possible_labels = collection_possible_labels(label_name)
            if sum(quotas[(source, label)] for label in possible_labels) <= 0:
                continue
            if label_name == "urlscan_live" and args.limit_per_collection > 0:
                for sampled_label in sorted(possible_labels):
                    sample_quota = int(quotas[(source, sampled_label)])
                    if sample_quota <= 0:
                        continue
                    for doc in sample_urlscan_documents(
                        client,
                        db_name,
                        collection_name,
                        args,
                        sampled_label,
                        sample_quota,
                    ):
                        record = build_input_record(doc, args, sampled_label, source)
                        output.write(
                            json.dumps(
                                record,
                                ensure_ascii=False,
                                separators=(",", ":"),
                            )
                        )
                        output.write("\n")
                        written += 1
                        quotas[(source, sampled_label)] -= 1
                if sum(quotas.values()) <= 0:
                    break
                continue
            for doc in iter_documents(client, db_name, collection_name, args):
                record_label = record_label_for_source(label_name, doc)
                if record_label is None:
                    continue
                if quotas[(source, record_label)] <= 0:
                    if sum(quotas[(source, label)] for label in possible_labels) <= 0:
                        break
                    continue
                record = build_input_record(doc, args, record_label, source)
                output.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
                output.write("\n")
                written += 1
                quotas[(source, record_label)] -= 1
                if sum(quotas.values()) <= 0:
                    break
            if sum(quotas.values()) <= 0:
                break

        if close_output:
            print(f"Wrote {written} JSONL record(s) to {args.output}", file=sys.stderr)
        return 0
    finally:
        if close_output:
            output.close()
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
