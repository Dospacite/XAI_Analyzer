#!/usr/bin/env python3
from __future__ import annotations

from typing import Any

from . import common as c


def redirect_urls(normalized_doc: dict[str, Any]) -> list[str]:
    urls: list[str] = []
    for item in (normalized_doc.get("metadata") or {}).get("redirect_history") or []:
        if isinstance(item, dict) and item.get("url"):
            urls.append(str(item["url"]))
    return urls


def extract(normalized_doc: dict[str, Any]) -> dict[str, Any]:
    requested_url, final_url = c.normalized_urls(normalized_doc)
    redirects = redirect_urls(normalized_doc)
    chain = [requested_url, *redirects, final_url]
    domains = [c.registrable_domain(url) for url in chain if url]
    domain_changes = sum(1 for before, after in zip(domains, domains[1:]) if before != after)
    requested = c.parse_url(requested_url)
    final = c.parse_url(final_url)
    return {
        "metadata.redirect_count": len(redirects),
        "metadata.redirect_domain_change_count": domain_changes,
        "metadata.final_url_changed": 1 if requested_url and final_url and requested_url != final_url else 0,
        "metadata.final_host_changed": 1 if c.hostname(requested_url) != c.hostname(final_url) else 0,
        "metadata.final_scheme_changed": 1 if requested.scheme.lower() != final.scheme.lower() else 0,
    }
