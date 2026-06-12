#!/usr/bin/env python3
"""Extract classical HTML/URL features directly from raw MongoDB documents.

The output is JSONL. Each line contains document metadata plus a `features`
object with the requested feature family values:

  {"id":"...","label":"phishing","source":"phishing_db.website_content",
   "url":"https://...","final_url":"https://...","features":{...}}

Binary features are encoded as 0/1 integers to keep the output friendly for
downstream dataframe-based ML workflows. Count/length features are integers.
`whole_url_string` is emitted as the resolved final URL string when available.

This extractor uses only raw HTML and stored fetch metadata. Features that
depend on runtime JavaScript execution are approximated conservatively from the
static page source, for example:

- `has_alert_window`: detects `alert(...)` patterns in the stored HTML/scripts.
- `has_redirection`: detects observed redirect history, meta refresh, or common
  static JavaScript redirect patterns.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urljoin, urlparse

from dotenv import load_dotenv
from parsel import Selector, SelectorList
from pymongo import MongoClient
import tldextract

try:
    from tqdm import tqdm
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    def tqdm(iterable=None, **_: Any):
        return iterable


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

TEXT_INPUT_TYPES = {"", "text", "search", "url", "tel", "number"}
BUTTON_INPUT_TYPES = {"button", "reset", "submit"}
GENERIC_HOST_LABELS = {
    "app",
    "auth",
    "cdn",
    "home",
    "login",
    "mail",
    "m",
    "portal",
    "secure",
    "signin",
    "signon",
    "support",
    "web",
    "www",
}
HIDDEN_STYLE_MARKERS = (
    "display:none",
    "visibility:hidden",
    "opacity:0",
)

ALERT_RE = re.compile(r"\b(?:window\s*\.\s*)?alert\s*\(", re.I)
JS_REDIRECT_RE = re.compile(
    r"""
    (?:
        \b(?:window\s*\.\s*|document\s*\.\s*)?location(?:\s*\.\s*(?:href|replace|assign))?\s*=
        |
        \b(?:window\s*\.\s*)?location\s*\.\s*(?:replace|assign)\s*\(
    )
    """,
    re.I | re.X,
)
EMAIL_HINT_RE = re.compile(r"\bemail\b", re.I)
PASSWORD_HINT_RE = re.compile(r"\bpass(word)?\b", re.I)

extract_tld = tldextract.TLDExtract(suffix_list_urls=())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        choices=["all", "phishing", "benign", "urlscan_live"],
        default="all",
        help="Mongo source to read.",
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
        "--limit-per-label",
        type=int,
        default=0,
        help="Maximum emitted rows per output label. 0 means no limit.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Mongo cursor batch size.",
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
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if max_chars is not None and len(text) > max_chars:
        return text[:max_chars].rstrip()
    return text


def optional_text(value: Any, max_chars: int | None = None) -> str | None:
    text = collapse_ws(value, max_chars)
    return text or None


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


def tag_attr(tag: Any, name: str) -> str | None:
    try:
        value = tag.root.get(name)
    except (AttributeError, TypeError, ValueError):
        value = None
    if isinstance(value, list):
        value = " ".join(str(item) for item in value)
    return optional_text(value, 2000)


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


def selector_text(selector: Any, max_chars: int | None = None) -> str | None:
    try:
        text_parts = selector.xpath(
            ".//text()[not(ancestor::script) and not(ancestor::style) "
            "and not(ancestor::noscript) and not(ancestor::svg) "
            "and not(ancestor::template)]"
        ).getall()
    except (AttributeError, TypeError, ValueError):
        text_parts = []
    return optional_text(" ".join(text_parts), max_chars)


def visible_text_from_selector(selector: Selector) -> str:
    body = css_select(selector, "body")
    body_text = selector_text(body)
    if body_text:
        return body_text
    return selector_text(selector) or ""


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
    return {
        "_id": doc.get("_id"),
        "url": optional_text(task.get("url"), 2000),
        "title": optional_text(page.get("title"), 300),
        "html": decode_html_blob(dom.get("data")),
        "error": None,
        "metadata": metadata,
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


def open_output(path: Path | None):
    if path is None:
        return sys.stdout
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.open("w", encoding="utf-8")


def lowercase_attr(tag: Any, name: str) -> str:
    return (tag_attr(tag, name) or "").strip().lower()


def css_count(selector: Selector, query: str) -> int:
    return len(css_select(selector, query))


def normalized_title(selector: Selector, doc_title: str | None) -> str:
    title_text = selector.css("title::text").get()
    return optional_text(title_text, 300) or optional_text(doc_title, 300) or ""


def input_type(tag: Any) -> str:
    return lowercase_attr(tag, "type")


def is_text_input(tag: Any) -> bool:
    return input_type(tag) in TEXT_INPUT_TYPES


def textish_attrs(tag: Any) -> str:
    return " ".join(
        filter(
            None,
            [
                tag_attr(tag, "name"),
                tag_attr(tag, "id"),
                tag_attr(tag, "placeholder"),
                tag_attr(tag, "autocomplete"),
                tag_attr(tag, "aria-label"),
            ],
        )
    )


def is_email_input(tag: Any) -> bool:
    if input_type(tag) == "email":
        return True
    return bool(EMAIL_HINT_RE.search(textish_attrs(tag)))


def is_password_input(tag: Any) -> bool:
    if input_type(tag) == "password":
        return True
    return bool(PASSWORD_HINT_RE.search(textish_attrs(tag)))


def is_submit_button(tag: Any) -> bool:
    tag_name = getattr(getattr(tag, "root", None), "tag", "").lower()
    tag_type = input_type(tag)
    if tag_name == "button":
        return tag_type in {"", "submit"}
    return tag_name == "input" and tag_type == "submit"


def registrable_domain(url: str) -> str:
    host = hostname(url)
    if not host:
        return ""
    extracted = extract_tld(host)
    if extracted.domain and extracted.suffix:
        return f"{extracted.domain}.{extracted.suffix}".lower()
    return host


def hostname(url: str) -> str:
    try:
        return (urlparse(url).hostname or "").lower().strip(".")
    except ValueError:
        return ""


def safe_urljoin(base_url: str, href: str) -> str:
    if not base_url:
        return href
    if base_url.count("[") != base_url.count("]"):
        return href
    try:
        urlparse(base_url)
    except ValueError:
        return href
    try:
        return urljoin(base_url, href)
    except ValueError:
        return href


def classify_link_target(base_url: str, href: str) -> str | None:
    href = (href or "").strip()
    if not href:
        return None

    href_l = href.lower()
    if href_l.startswith(("javascript:", "mailto:", "tel:", "data:")):
        return None

    resolved = safe_urljoin(base_url, href)
    resolved_host = hostname(resolved)
    base_host = hostname(base_url)
    if not resolved_host:
        return "internal"

    resolved_domain = registrable_domain(resolved)
    base_domain = registrable_domain(base_url)
    if resolved_domain and base_domain and resolved_domain == base_domain:
        return "internal"
    if resolved_host == base_host:
        return "internal"
    return "external"


def style_is_hidden(style: str) -> bool:
    style_l = re.sub(r"\s+", "", style.lower())
    if any(marker in style_l for marker in HIDDEN_STYLE_MARKERS):
        return True
    return "width:0" in style_l and "height:0" in style_l


def has_hidden_element(selector: Selector) -> int:
    for element in selector.xpath("//*"):
        if lowercase_attr(element, "hidden"):
            return 1
        if lowercase_attr(element, "aria-hidden") == "true":
            return 1
        if input_type(element) == "hidden":
            return 1
        style = tag_attr(element, "style")
        if style and style_is_hidden(style):
            return 1
    return 0


def hostname_brand_tokens(url: str) -> list[str]:
    host = hostname(url)
    if not host:
        return []

    tokens: list[str] = []
    for label in host.split("."):
        normalized = re.sub(r"[^a-z0-9]+", "", label.lower())
        if len(normalized) < 3 or normalized in GENERIC_HOST_LABELS:
            continue
        tokens.append(normalized)

    domain = extract_tld(host).domain.lower()
    normalized_domain = re.sub(r"[^a-z0-9]+", "", domain)
    if len(normalized_domain) >= 3 and normalized_domain not in GENERIC_HOST_LABELS:
        tokens.append(normalized_domain)

    deduped: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        if token in seen:
            continue
        seen.add(token)
        deduped.append(token)
    return deduped


def title_url_brand_consistency(title: str, page_url: str) -> int:
    normalized_title_text = re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()
    if not normalized_title_text:
        return 0

    candidate_tokens = hostname_brand_tokens(page_url)
    if not candidate_tokens:
        return 0

    for token in candidate_tokens:
        if token in normalized_title_text:
            return 1
    return 0


def has_meta_refresh(selector: Selector) -> bool:
    for tag in css_select(selector, "meta"):
        if lowercase_attr(tag, "http-equiv") != "refresh":
            continue
        return True
    return False


def detect_redirection(
    normalized_doc: dict[str, Any],
    selector: Selector,
    html: str,
    page_url: str,
    final_url: str,
) -> int:
    metadata = normalized_doc.get("metadata") or {}
    if metadata.get("redirect_history"):
        return 1
    if page_url and final_url and page_url != final_url:
        return 1
    if has_meta_refresh(selector):
        return 1
    if JS_REDIRECT_RE.search(html or ""):
        return 1
    return 0


def detect_alert_window(html: str) -> int:
    return 1 if ALERT_RE.search(html or "") else 0


def extract_features_from_document(doc: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
    normalized_doc = normalize_document(doc)
    html = normalized_doc.get("html") or ""
    selector = html_selector_from_text(html)
    metadata = normalized_doc.get("metadata") or {}
    page_url = collapse_ws(normalized_doc.get("url") or metadata.get("url"), 2000)
    final_url = collapse_ws(
        metadata.get("final_url") or normalized_doc.get("url") or metadata.get("url"),
        2000,
    )
    title = normalized_title(selector, normalized_doc.get("title"))
    visible_text = visible_text_from_selector(selector)

    input_tags = css_select(selector, "input")
    button_tags = css_select(selector, "button")
    anchor_tags = css_select(selector, "a")

    text_input_count = sum(1 for tag in input_tags if is_text_input(tag))
    submit_count = sum(1 for tag in button_tags if is_submit_button(tag))
    submit_count += sum(1 for tag in input_tags if is_submit_button(tag))
    password_count = sum(1 for tag in input_tags if is_password_input(tag))
    email_count = sum(1 for tag in input_tags if is_email_input(tag))

    internal_links = 0
    external_links = 0
    href_count = 0
    for anchor in anchor_tags:
        href = tag_attr(anchor, "href")
        if href is None:
            continue
        href_count += 1
        link_class = classify_link_target(final_url or page_url, href)
        if link_class == "internal":
            internal_links += 1
        elif link_class == "external":
            external_links += 1

    clickable_button_count = len(selector.xpath(
        "//button"
        " | //input[translate(@type,'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz')='submit'"
        "          or translate(@type,'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz')='button'"
        "          or translate(@type,'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz')='reset']"
        " | //*[@role='button' and not(self::button)]"
    ))

    features = {
        "has_title": 1 if title else 0,
        "has_input": 1 if input_tags else 0,
        "has_text_input": 1 if text_input_count > 0 else 0,
        "has_submit": 1 if submit_count > 0 else 0,
        "number_of_internal_links": internal_links,
        "number_of_external_links": external_links,
        "has_password": 1 if password_count > 0 else 0,
        "has_email_input": 1 if email_count > 0 else 0,
        "has_hidden_element": has_hidden_element(selector),
        "has_audio": 1 if css_count(selector, "audio") > 0 else 0,
        "has_video": 1 if css_count(selector, "video") > 0 else 0,
        "number_of_inputs": len(input_tags),
        "number_of_buttons": len(button_tags),
        "number_of_images": css_count(selector, "img"),
        "has_image": 1 if css_count(selector, "img") > 0 else 0,
        "number_of_options": css_count(selector, "option"),
        "number_of_lists": len(selector.xpath("//ul | //ol | //dl")),
        "number_of_th": css_count(selector, "th"),
        "number_of_tr": css_count(selector, "tr"),
        "number_of_tables": css_count(selector, "table"),
        "number_of_href": href_count,
        "number_of_paragraphs": css_count(selector, "p"),
        "number_of_scripts": css_count(selector, "script"),
        "number_of_clickable_buttons": clickable_button_count,
        "number_of_tags": len(selector.xpath("//*")),
        "number_of_divs": css_count(selector, "div"),
        "number_of_figures": css_count(selector, "figure"),
        "has_footer": 1 if css_count(selector, "footer") > 0 else 0,
        "has_form": 1 if css_count(selector, "form") > 0 else 0,
        "has_text_area": 1 if css_count(selector, "textarea") > 0 else 0,
        "has_iframe": 1 if css_count(selector, "iframe") > 0 else 0,
        "number_of_meta": css_count(selector, "meta"),
        "has_navigation": 1 if css_count(selector, "nav") > 0 else 0,
        "has_object": 1 if css_count(selector, "object") > 0 else 0,
        "has_picture": 1 if css_count(selector, "picture") > 0 else 0,
        "number_of_sources": css_count(selector, "source"),
        "number_of_spans": css_count(selector, "span"),
        "length_of_text": len(visible_text),
        "length_of_title": len(title),
        "has_h1": 1 if css_count(selector, "h1") > 0 else 0,
        "has_h2": 1 if css_count(selector, "h2") > 0 else 0,
        "has_h3": 1 if css_count(selector, "h3") > 0 else 0,
        "has_alert_window": detect_alert_window(html),
        "has_redirection": detect_redirection(
            normalized_doc,
            selector,
            html,
            page_url,
            final_url,
        ),
        "consistency_between_title_and_url_brand": title_url_brand_consistency(
            title,
            final_url or page_url,
        ),
        "whole_url_string": final_url or page_url,
    }
    return page_url, final_url, features


def collection_possible_labels(label_name: str) -> set[str]:
    if label_name == "urlscan_live":
        return {"phishing", "benign"}
    return {label_name}


def selected_output_labels(source: str) -> set[str]:
    if source == "all":
        return {"phishing", "benign"}
    if source == "urlscan_live":
        return {"phishing", "benign"}
    return {source}


def iter_documents(
    client: MongoClient,
    db_name: str,
    collection_name: str,
    args: argparse.Namespace,
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


def record_label_for_source(label_name: str, doc: dict[str, Any]) -> str | None:
    if label_name != "urlscan_live":
        return label_name
    return urlscan_label(doc)


def build_output_record(
    doc: dict[str, Any],
    label: str,
    source: str,
) -> dict[str, Any]:
    normalized_doc = normalize_document(doc)
    page_url, final_url, features = extract_features_from_document(normalized_doc)
    return {
        "id": stable_record_id(normalized_doc),
        "label": label,
        "source": source,
        "url": page_url,
        "final_url": final_url,
        "features": features,
    }


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
        total_written = 0
        written_by_label = {label: 0 for label in selected_output_labels(args.source)}
        for label_name, db_name, collection_name in selected_collections(args.source):
            source = f"{db_name}.{collection_name}"
            for doc in iter_documents(client, db_name, collection_name, args):
                possible_labels = collection_possible_labels(label_name)
                if args.limit_per_label > 0 and all(
                    written_by_label.get(candidate_label, 0) >= args.limit_per_label
                    for candidate_label in possible_labels
                ):
                    break

                label = record_label_for_source(label_name, doc)
                if label not in collection_possible_labels(label_name):
                    continue
                if (
                    args.limit_per_label > 0
                    and written_by_label.get(label, 0) >= args.limit_per_label
                ):
                    continue

                record = build_output_record(doc, label, source)
                output.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
                output.write("\n")
                total_written += 1
                written_by_label[label] = written_by_label.get(label, 0) + 1

        if close_output:
            print(
                f"Wrote {total_written} JSONL record(s) to {args.output}",
                file=sys.stderr,
            )
        return 0
    finally:
        if close_output:
            output.close()


if __name__ == "__main__":
    raise SystemExit(main())
