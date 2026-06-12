#!/usr/bin/env python3
from __future__ import annotations

from typing import Any

from . import common as c
from . import html_features, metadata_features, url_features


def extract_feature_groups(doc: dict[str, Any]) -> dict[str, dict[str, Any]]:
    normalized_doc = c.normalize_document(doc)
    return {
        "URL": url_features.extract(normalized_doc),
        "HTML": html_features.extract(normalized_doc),
        "METADATA": metadata_features.extract(normalized_doc),
    }


def flatten_feature_groups(groups: dict[str, dict[str, Any]]) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for group in groups.values():
        flat.update(group)
    return flat


def build_feature_record(
    doc: dict[str, Any],
    db_name: str,
    collection_name: str,
    label: str | None = None,
    label_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    normalized_doc = c.normalize_document(doc)
    requested_url, final_url = c.normalized_urls(normalized_doc)
    groups = {
        "URL": url_features.extract(normalized_doc),
        "HTML": html_features.extract(normalized_doc),
        "METADATA": metadata_features.extract(normalized_doc),
    }
    return {
        "id": c.stable_record_id(normalized_doc),
        "label": label,
        "label_info": label_info or {},
        "db": db_name,
        "collection": collection_name,
        "url": requested_url,
        "final_url": final_url,
        "title": c.title_from_doc_or_html(
            normalized_doc,
            c.html_selector(normalized_doc.get("html") or ""),
        ),
        "feature_groups": groups,
        "features": flatten_feature_groups(groups),
    }
