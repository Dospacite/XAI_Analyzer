#!/usr/bin/env python3
from __future__ import annotations

import re
from typing import Any
from urllib.parse import unquote

from . import common as c


def extract(normalized_doc: dict[str, Any]) -> dict[str, Any]:
    requested_url, final_url = c.normalized_urls(normalized_doc)
    parsed = c.parse_url(final_url)
    host = c.hostname(final_url)
    registered = c.registrable_domain(final_url)
    subdomain = c.subdomain_text(final_url)
    after_scheme = final_url.split("://", 1)[-1]
    token_lengths = c.token_lengths(final_url)
    digits = c.count_regex(r"[0-9]", final_url)
    path_segments = [segment for segment in (parsed.path or "").split("/") if segment]
    last_path_segment = path_segments[-1] if path_segments else ""
    extension_match = re.search(r"\.([A-Za-z0-9]{1,12})$", last_path_segment)
    file_extension = extension_match.group(1).lower() if extension_match else ""
    path_query = f"{parsed.path or ''}?{parsed.query or ''}"
    decoded_path_query = unquote(path_query).lower()
    return {
        "url.final_url_length": len(final_url),
        "url.requested_url_length": len(requested_url),
        "url.scheme_is_https": 1 if parsed.scheme.lower() == "https" else 0,
        "url.hostname_length": len(host),
        "url.registrable_domain_length": len(registered),
        "url.subdomain_length": len(subdomain),
        "url.path_length": len(parsed.path or ""),
        "url.query_length": len(parsed.query or ""),
        "url.fragment_length": len(parsed.fragment or ""),
        "url.path_segment_count": len(path_segments),
        "url.query_parameter_count": c.query_param_count(final_url),
        "url.subdomain_label_count": c.subdomain_label_count(final_url),
        "url.dot_count": final_url.count("."),
        "url.hostname_dot_count": host.count("."),
        "url.hyphen_count": final_url.count("-"),
        "url.registrable_domain_hyphen_count": registered.count("-"),
        "url.underscore_count": final_url.count("_"),
        "url.slash_count": final_url.count("/"),
        "url.extra_double_slash_count": after_scheme.count("//"),
        "url.question_mark_count": final_url.count("?"),
        "url.equals_sign_count": final_url.count("="),
        "url.ampersand_count": final_url.count("&"),
        "url.at_sign_count": final_url.count("@"),
        "url.percent_sign_count": final_url.count("%"),
        "url.hash_sign_count": final_url.count("#"),
        "url.tilde_count": final_url.count("~"),
        "url.plus_sign_count": final_url.count("+"),
        "url.asterisk_count": final_url.count("*"),
        "url.parenthesis_count": final_url.count("(") + final_url.count(")"),
        "url.square_bracket_count": final_url.count("[") + final_url.count("]"),
        "url.curly_bracket_count": final_url.count("{") + final_url.count("}"),
        "url.angle_bracket_count": final_url.count("<") + final_url.count(">"),
        "url.digit_count": digits,
        "url.hostname_digit_count": c.count_regex(r"[0-9]", host),
        "url.digit_ratio": digits / max(1, len(final_url)),
        "url.special_character_count": c.count_regex(r"[^A-Za-z0-9]", final_url),
        "url.token_count": len(token_lengths),
        "url.average_token_length": sum(token_lengths) / len(token_lengths) if token_lengths else 0.0,
        "url.longest_token_length": max(token_lengths) if token_lengths else 0,
        "url.shortest_token_length": min(token_lengths) if token_lengths else 0,
        "url.token_length_stddev": c.stddev(token_lengths),
        "url.host_is_ip_address": 1 if c.is_ip_hostname(final_url) else 0,
        "url.punycode_present": 1 if any(label.startswith("xn--") for label in host.split(".")) else 0,
        "url.explicit_port_present": 1 if parsed.port is not None else 0,
        "url.non_default_port_present": 1 if parsed.port is not None and parsed.port not in {80, 443} else 0,
        "url.https_token_in_hostname": 1 if "https" in host else 0,
        "url.has_file_extension": 1 if file_extension else 0,
        "url.path_or_query_contains_url": 1
        if re.search(r"https?://|https?%3a%2f%2f|www\.", decoded_path_query, flags=re.I)
        else 0,
        "url.character_entropy": c.entropy(final_url),
    }
