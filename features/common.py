#!/usr/bin/env python3
from __future__ import annotations

import gzip
import hashlib
import json
import math
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urljoin, urlparse

from parsel import Selector, SelectorList
import tldextract

try:
    from bson import ObjectId
except Exception:  # pragma: no cover - bson is provided by pymongo in normal use
    ObjectId = None  # type: ignore[assignment]


DEFAULT_COLLECTIONS = {
    "phishing": ("phishing_db", "website_content"),
    "benign": ("tranco", "websites"),
    "urlscan_live": ("urlscan", "live"),
}

COLLECTION_LABELS = {
    ("phishing_db", "website_content"): "phishing",
    ("tranco", "websites"): "benign",
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
HIDDEN_STYLE_MARKERS = ("display:none", "visibility:hidden", "opacity:0")

extract_tld = tldextract.TLDExtract(suffix_list_urls=())


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


def decode_html_blob(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (bytes, bytearray)):
        try:
            return gzip.decompress(value).decode("utf-8", errors="replace")
        except (OSError, EOFError, TypeError, ValueError):
            return bytes(value).decode("utf-8", errors="replace")
    return ""


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


def json_default(value: Any) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def extract_redirects_from_urlscan(doc: dict[str, Any]) -> list[dict[str, Any]]:
    requests = (((doc.get("urlscanresults") or {}).get("data") or {}).get("requests") or [])
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


def normalize_document(doc: dict[str, Any]) -> dict[str, Any]:
    """Normalize phishing/tranco/urlscan documents to common feature fields."""
    if isinstance(doc.get("html"), str):
        metadata = dict(doc.get("metadata") or {})
        return {
            "_id": doc.get("_id"),
            "url": optional_text(first_non_empty(doc.get("url"), metadata.get("url")), 2000),
            "title": optional_text(doc.get("title"), 300),
            "html": doc.get("html") or "",
            "error": doc.get("error"),
            "metadata": metadata,
        }

    task = first_non_empty(doc.get("task"), (doc.get("urlscanresults") or {}).get("task"))
    page = first_non_empty(doc.get("page"), (doc.get("urlscanresults") or {}).get("page"))
    task = task if isinstance(task, dict) else {}
    page = page if isinstance(page, dict) else {}
    dom = doc.get("dom") if isinstance(doc.get("dom"), dict) else {}
    metadata = {
        "url": optional_text(task.get("url"), 2000),
        "final_url": optional_text(first_non_empty(page.get("url"), task.get("url")), 2000),
        "redirect_history": extract_redirects_from_urlscan(doc),
        "status_code": safe_int(page.get("status")),
    }
    return {
        "_id": doc.get("_id"),
        "url": optional_text(task.get("url"), 2000),
        "title": optional_text(page.get("title"), 300),
        "html": decode_html_blob(dom.get("data")),
        "error": None,
        "metadata": metadata,
    }


def normalized_urls(normalized_doc: dict[str, Any]) -> tuple[str, str]:
    metadata = normalized_doc.get("metadata") or {}
    requested = collapse_ws(first_non_empty(normalized_doc.get("url"), metadata.get("url")), 2000)
    final = collapse_ws(first_non_empty(metadata.get("final_url"), requested), 2000)
    return requested, final


def normalized_status_code(normalized_doc: dict[str, Any]) -> int | None:
    status = (normalized_doc.get("metadata") or {}).get("status_code")
    return safe_int(status)


def safe_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def html_selector(html: str) -> Selector:
    selector = Selector(text=html or "", type="html")
    if selector.type != "html":
        return Selector(text=f"<!doctype html><html><body>{html or ''}</body></html>", type="html")
    return selector


def css_select(selector: Any, query: str) -> SelectorList:
    try:
        return selector.css(query)
    except (AttributeError, TypeError, ValueError):
        return SelectorList([])


def tag_attr(tag: Any, name: str) -> str | None:
    try:
        value = tag.root.get(name)
    except (AttributeError, TypeError, ValueError):
        value = None
    if isinstance(value, list):
        value = " ".join(str(item) for item in value)
    return optional_text(value, 2000)


def lowercase_attr(tag: Any, name: str) -> str:
    return (tag_attr(tag, name) or "").strip().lower()


def selector_text(selector: Any, max_chars: int | None = None) -> str:
    try:
        text_parts = selector.xpath(
            ".//text()[not(ancestor::script) and not(ancestor::style) "
            "and not(ancestor::noscript) and not(ancestor::svg) "
            "and not(ancestor::template)]"
        ).getall()
    except (AttributeError, TypeError, ValueError):
        text_parts = []
    return collapse_ws(" ".join(text_parts), max_chars)


def visible_text(selector: Selector) -> str:
    body = css_select(selector, "body")
    return selector_text(body) or selector_text(selector)


def title_from_doc_or_html(normalized_doc: dict[str, Any], selector: Selector) -> str:
    doc_title = optional_text(normalized_doc.get("title"), 300)
    if doc_title:
        return doc_title
    try:
        return optional_text(selector.css("title::text").get(), 300) or ""
    except (AttributeError, TypeError, ValueError):
        return ""


def parse_url(url: str):
    try:
        return urlparse(url or "")
    except ValueError:
        return urlparse("")


def hostname(url: str) -> str:
    try:
        return (urlparse(url or "").hostname or "").lower().strip(".")
    except ValueError:
        return ""


def tld_parts(url_or_host: str):
    host = hostname(url_or_host) or (url_or_host or "").lower().strip(".")
    return extract_tld(host)


def registrable_domain(url_or_host: str) -> str:
    extracted = tld_parts(url_or_host)
    if extracted.domain and extracted.suffix:
        return f"{extracted.domain}.{extracted.suffix}".lower()
    if extracted.domain:
        return extracted.domain.lower()
    return hostname(url_or_host)


def registered_domain_token(url_or_host: str) -> str:
    return tld_parts(url_or_host).domain.lower()


def subdomain_text(url_or_host: str) -> str:
    return tld_parts(url_or_host).subdomain.lower()


def subdomain_label_count(url_or_host: str) -> int:
    subdomain = subdomain_text(url_or_host)
    return len([label for label in subdomain.split(".") if label])


def host_label_count(url_or_host: str) -> int:
    host = hostname(url_or_host)
    return len([label for label in host.split(".") if label])


def is_ip_hostname(url: str) -> bool:
    import ipaddress

    host = hostname(url)
    if not host:
        return False
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        return False


def safe_urljoin(base_url: str, href: str) -> str:
    try:
        return urljoin(base_url or "", href or "")
    except ValueError:
        return href or ""


def classify_link_target(base_url: str, reference: str) -> str | None:
    reference = (reference or "").strip()
    if not reference:
        return None
    lowered = reference.lower()
    if lowered in {"#", "/", "about:blank", "javascript:", "javascript:void(0)", "javascript:void(0);"}:
        return None
    if lowered.startswith(("javascript:", "mailto:", "tel:", "data:")):
        return None
    resolved = safe_urljoin(base_url, reference)
    resolved_host = hostname(resolved)
    if not resolved_host:
        return "internal"
    resolved_domain = registrable_domain(resolved)
    base_domain = registrable_domain(base_url)
    if resolved_domain and base_domain and resolved_domain == base_domain:
        return "internal"
    return "external"


def is_null_reference(reference: str | None) -> bool:
    text = (reference or "").strip().lower()
    return (
        text == ""
        or text == "#"
        or text.startswith("#")
        or text in {"javascript:", "javascript:void(0)", "javascript:void(0);", "about:blank"}
    )


def token_lengths(value: str) -> list[int]:
    return [len(token) for token in re.split(r"[^A-Za-z0-9]+", value or "") if token]


def stddev(values: list[int | float]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def entropy(value: str) -> float:
    if not value:
        return 0.0
    counts = Counter(value)
    total = len(value)
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


def count_regex(pattern: str, value: str, flags: int = 0) -> int:
    return len(re.findall(pattern, value or "", flags=flags))


def query_param_count(url: str) -> int:
    try:
        return len(parse_qsl(urlparse(url).query, keep_blank_values=True))
    except ValueError:
        return 0


def mongo_html_query() -> dict[str, Any]:
    return {
        "$or": [
            {"html": {"$exists": True, "$type": "string", "$ne": ""}},
            {"dom.data": {"$exists": True, "$ne": None}},
        ]
    }


def mongo_feature_projection() -> dict[str, int]:
    return {
        "_id": 1,
        "url": 1,
        "title": 1,
        "html": 1,
        "error": 1,
        "metadata.url": 1,
        "metadata.final_url": 1,
        "metadata.status_code": 1,
        "metadata.redirect_history": 1,
        "task.url": 1,
        "page.url": 1,
        "page.title": 1,
        "page.status": 1,
        "dom.data": 1,
        "scans.T0.virustotal": 1,
        "scans.T0.google_safe_browsing": 1,
        "scans.T7.virustotal": 1,
        "scans.T7.google_safe_browsing": 1,
        "urlscanresults.task.url": 1,
        "urlscanresults.page.url": 1,
        "urlscanresults.page.title": 1,
        "urlscanresults.page.status": 1,
        "urlscanresults.data.requests.requests.redirectResponse": 1,
    }


def vt_marked_phishing(virustotal: dict[str, Any]) -> bool:
    if safe_int(virustotal.get("malicious_engine_count")) and safe_int(virustotal.get("malicious_engine_count")) > 0:
        return True
    if safe_int(virustotal.get("suspicious_engine_count")) and safe_int(virustotal.get("suspicious_engine_count")) > 0:
        return True
    stats = virustotal.get("stats") if isinstance(virustotal.get("stats"), dict) else {}
    if safe_int(stats.get("malicious")) and safe_int(stats.get("malicious")) > 0:
        return True
    if safe_int(stats.get("suspicious")) and safe_int(stats.get("suspicious")) > 0:
        return True
    for result in virustotal.get("results") or virustotal.get("analysis_results") or []:
        if not isinstance(result, dict):
            continue
        category = collapse_ws(result.get("category"), 100).lower()
        if category in {"malicious", "suspicious", "phishing", "malware"}:
            return True
    return False


def vt_has_verdict(virustotal: Any) -> bool:
    if not isinstance(virustotal, dict) or not virustotal:
        return False
    if isinstance(virustotal.get("stats"), dict):
        return True
    if virustotal.get("malicious_engine_count") is not None:
        return True
    if virustotal.get("suspicious_engine_count") is not None:
        return True
    return False


def gsb_marked_phishing(gsb: dict[str, Any]) -> bool:
    status = collapse_ws(gsb.get("status"), 100).lower()
    if gsb.get("matched") is True:
        return True
    if safe_int(gsb.get("match_count")) and safe_int(gsb.get("match_count")) > 0:
        return True
    if status in {"malicious", "phishing", "unsafe", "threat"}:
        return True
    if gsb.get("matches"):
        return True
    return False


def gsb_has_verdict(gsb: Any) -> bool:
    if not isinstance(gsb, dict) or not gsb:
        return False
    return any(key in gsb for key in ("matched", "status", "match_count", "matches"))


def urlscan_scan_label(scan: Any) -> tuple[str | None, dict[str, Any]]:
    if not isinstance(scan, dict) or not scan:
        return None, {}
    vt = scan.get("virustotal")
    gsb = scan.get("google_safe_browsing")
    has_vt = vt_has_verdict(vt)
    has_gsb = gsb_has_verdict(gsb)
    if not has_vt or not has_gsb:
        return None, {}
    vt_phishing = vt_marked_phishing(vt) if isinstance(vt, dict) else False
    gsb_phishing = gsb_marked_phishing(gsb) if isinstance(gsb, dict) else False
    label = "phishing" if vt_phishing or gsb_phishing else "benign"
    evidence = {
        "scan": None,
        "vt_has_verdict": has_vt,
        "vt_marked_phishing": vt_phishing,
        "vt_malicious_engine_count": safe_int((vt or {}).get("malicious_engine_count")) if isinstance(vt, dict) else None,
        "vt_suspicious_engine_count": safe_int((vt or {}).get("suspicious_engine_count")) if isinstance(vt, dict) else None,
        "gsb_has_verdict": has_gsb,
        "gsb_marked_phishing": gsb_phishing,
        "gsb_status": collapse_ws((gsb or {}).get("status"), 100).lower() if isinstance(gsb, dict) else "",
        "gsb_matched": (gsb or {}).get("matched") if isinstance(gsb, dict) else None,
    }
    return label, evidence


def infer_dataset_label(db_name: str, collection_name: str, doc: dict[str, Any]) -> tuple[str | None, dict[str, Any]]:
    fixed = COLLECTION_LABELS.get((db_name, collection_name))
    if fixed:
        return fixed, {"rule": f"{db_name}.{collection_name}", "latest_scan": None}
    if (db_name, collection_name) != ("urlscan", "live"):
        return None, {"rule": "unknown_collection", "latest_scan": None}

    scans = doc.get("scans") if isinstance(doc.get("scans"), dict) else {}
    labels: dict[str, str | None] = {}
    evidences: dict[str, dict[str, Any]] = {}
    for scan_name in ("T0", "T7"):
        label, evidence = urlscan_scan_label(scans.get(scan_name))
        labels[scan_name] = label
        if evidence:
            evidence["scan"] = scan_name
            evidences[scan_name] = evidence

    latest_scan = "T7" if labels.get("T7") else "T0" if labels.get("T0") else None
    if latest_scan is None:
        return None, {"rule": "urlscan_latest_t7_else_t0", "latest_scan": None, "scan_labels": labels}
    return labels[latest_scan], {
        "rule": "urlscan_latest_t7_else_t0_any_vt_or_gsb_hit_is_phishing",
        "latest_scan": latest_scan,
        "scan_labels": labels,
        "latest_evidence": evidences.get(latest_scan, {}),
    }


def build_lookup_query(mongo_id: str | None, website_url: str | None) -> dict[str, Any]:
    if mongo_id:
        candidates: list[Any] = [mongo_id]
        if ObjectId is not None:
            try:
                candidates.append(ObjectId(mongo_id))
            except Exception:
                pass
        return {"_id": {"$in": candidates}}
    if website_url:
        return {
            "$or": [
                {"url": website_url},
                {"metadata.url": website_url},
                {"metadata.final_url": website_url},
                {"task.url": website_url},
                {"page.url": website_url},
                {"urlscanresults.task.url": website_url},
                {"urlscanresults.page.url": website_url},
            ]
        }
    raise ValueError("Either mongo_id or website_url is required.")


def write_json_line(handle: Any, payload: dict[str, Any]) -> None:
    handle.write(json.dumps(payload, ensure_ascii=False, default=json_default, separators=(",", ":")))
    handle.write("\n")
