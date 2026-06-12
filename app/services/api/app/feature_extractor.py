from __future__ import annotations

import ipaddress
import math
import re
from collections import Counter
from typing import Any
from urllib.parse import parse_qsl, unquote, urljoin, urlsplit

import tldextract
from bs4 import BeautifulSoup, Tag

_extract_domain = tldextract.TLDExtract(suffix_list_urls=(), cache_dir=None)
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_CREDENTIAL_RE = re.compile(
    r"\b(login|log in|sign in|signin|username|user name|password|passcode|credential|email address)\b",
    re.IGNORECASE,
)
_PRIVACY_RE = re.compile(r"\b(privacy|terms(?: of (?:service|use))?|legal|cookies?)\b", re.IGNORECASE)
_JS_REDIRECT_RE = re.compile(
    r"(?:window|document|top|self)\s*\.\s*location(?:\s*\.href)?\s*=|location\s*\.\s*replace\s*\(",
    re.IGNORECASE,
)
_RIGHT_CLICK_RE = re.compile(
    r"(?:contextmenu|oncontextmenu)\b|event\s*\.\s*button\s*==?\s*2|return\s+false",
    re.IGNORECASE,
)
_HIDDEN_STYLE_RE = re.compile(
    r"display\s*:\s*none|visibility\s*:\s*hidden|opacity\s*:\s*0(?:\D|$)|left\s*:\s*-\d{3,}px",
    re.IGNORECASE,
)
_URL_IN_URL_RE = re.compile(r"https?%3a|https?://", re.IGNORECASE)
_PLACEHOLDER_HREFS = {"", "#", "#!", "javascript:void(0)", "javascript:void(0);", "javascript:;", "about:blank"}


def _entropy(value: str) -> float:
    if not value:
        return 0.0
    counts = Counter(value)
    length = len(value)
    return -sum((count / length) * math.log2(count / length) for count in counts.values())


def _domain_parts(hostname: str) -> tuple[str, str, str]:
    result = _extract_domain(hostname or "")
    registered = ".".join(part for part in (result.domain, result.suffix) if part)
    return result.subdomain or "", result.domain or "", registered


def _registrable_domain(hostname: str) -> str:
    return _domain_parts(hostname)[2] or hostname.lower().strip(".")


def _safe_port(parsed: Any) -> int | None:
    try:
        return parsed.port
    except ValueError:
        return None


def _is_external(reference: str, base_url: str) -> bool:
    reference = (reference or "").strip()
    if not reference or reference.startswith(("data:", "blob:", "javascript:", "mailto:", "tel:", "#")):
        return False
    absolute = urlsplit(urljoin(base_url, reference))
    base = urlsplit(base_url)
    if not absolute.hostname:
        return False
    return _registrable_domain(absolute.hostname) != _registrable_domain(base.hostname or "")


def _visible_text(soup: BeautifulSoup) -> str:
    chunks: list[str] = []
    for node in soup.find_all(string=True):
        parent = node.parent
        if not isinstance(parent, Tag) or parent.name in {"script", "style", "noscript", "template", "svg"}:
            continue
        if parent.has_attr("hidden") or _HIDDEN_STYLE_RE.search(str(parent.get("style", ""))):
            continue
        text = " ".join(str(node).split())
        if text:
            chunks.append(text)
    return " ".join(chunks)


def _tokens(value: str) -> list[str]:
    return [token.lower() for token in _TOKEN_RE.findall(value or "") if token]


def extract_features(document: dict[str, Any]) -> dict[str, int | float | str]:
    requested_url = str(document.get("url") or "")
    metadata = document.get("metadata") or {}
    final_url = str(metadata.get("final_url") or requested_url)
    html = str(document.get("html") or "")
    title = str(document.get("title") or "")

    requested = urlsplit(requested_url)
    final = urlsplit(final_url)
    hostname = final.hostname or requested.hostname or ""
    subdomain, domain_token, registered_domain = _domain_parts(hostname)
    path = final.path or ""
    query = final.query or ""
    fragment = final.fragment or ""
    url_value = final_url

    url_tokens = _tokens(" ".join((hostname, path, query)))
    token_lengths = [len(token) for token in url_tokens]
    mean_token = sum(token_lengths) / len(token_lengths) if token_lengths else 0.0
    token_variance = (
        sum((length - mean_token) ** 2 for length in token_lengths) / len(token_lengths)
        if token_lengths
        else 0.0
    )
    digit_count = sum(character.isdigit() for character in url_value)
    special_count = sum(not character.isalnum() for character in url_value)
    explicit_port = _safe_port(final)
    default_port = 443 if final.scheme == "https" else 80 if final.scheme == "http" else None
    try:
        host_is_ip = int(ipaddress.ip_address(hostname) is not None)
    except ValueError:
        host_is_ip = 0

    suffix = path.rsplit("/", 1)[-1]
    file_extension = suffix.rsplit(".", 1)[-1].lower() if "." in suffix else ""
    has_file_extension = int(bool(file_extension and 1 <= len(file_extension) <= 10))

    features: dict[str, int | float | str] = {
        "url.requested_url_length": len(requested_url),
        "url.final_url_length": len(final_url),
        "url.hostname_length": len(hostname),
        "url.hostname_digit_count": sum(character.isdigit() for character in hostname),
        "url.registrable_domain_length": len(registered_domain),
        "url.subdomain_length": len(subdomain),
        "url.path_length": len(path),
        "url.query_length": len(query),
        "url.fragment_length": len(fragment),
        "url.digit_count": digit_count,
        "url.digit_ratio": digit_count / max(len(url_value), 1),
        "url.character_entropy": _entropy(url_value),
        "url.special_character_count": special_count,
        "url.dot_count": url_value.count("."),
        "url.hostname_dot_count": hostname.count("."),
        "url.hyphen_count": url_value.count("-"),
        "url.registrable_domain_hyphen_count": registered_domain.count("-"),
        "url.at_sign_count": url_value.count("@"),
        "url.percent_sign_count": url_value.count("%"),
        "url.equals_sign_count": url_value.count("="),
        "url.underscore_count": url_value.count("_"),
        "url.ampersand_count": url_value.count("&"),
        "url.hash_sign_count": url_value.count("#"),
        "url.question_mark_count": url_value.count("?"),
        "url.slash_count": url_value.count("/"),
        "url.parenthesis_count": url_value.count("(") + url_value.count(")"),
        "url.square_bracket_count": url_value.count("[") + url_value.count("]"),
        "url.curly_bracket_count": url_value.count("{") + url_value.count("}"),
        "url.angle_bracket_count": url_value.count("<") + url_value.count(">"),
        "url.asterisk_count": url_value.count("*"),
        "url.plus_sign_count": url_value.count("+"),
        "url.tilde_count": url_value.count("~"),
        "url.extra_double_slash_count": max(url_value.count("//") - 1, 0),
        "url.token_count": len(url_tokens),
        "url.average_token_length": mean_token,
        "url.longest_token_length": max(token_lengths, default=0),
        "url.shortest_token_length": min(token_lengths, default=0),
        "url.token_length_stddev": math.sqrt(token_variance),
        "url.path_segment_count": len([part for part in path.split("/") if part]),
        "url.query_parameter_count": len(parse_qsl(query, keep_blank_values=True)),
        "url.subdomain_label_count": len([part for part in subdomain.split(".") if part]),
        "url.scheme_is_https": int(final.scheme.lower() == "https"),
        "url.host_is_ip_address": host_is_ip,
        "url.punycode_present": int("xn--" in hostname.lower()),
        "url.https_token_in_hostname": int("https" in hostname.lower()),
        "url.explicit_port_present": int(explicit_port is not None),
        "url.non_default_port_present": int(explicit_port is not None and explicit_port != default_port),
        "url.path_or_query_contains_url": int(bool(_URL_IN_URL_RE.search(unquote(f"{path}?{query}")))),
        "url.has_file_extension": has_file_extension,
        "url.file_extension": file_extension,
    }

    soup = BeautifulSoup(html, "lxml")
    if not title:
        title_node = soup.find("title")
        title = title_node.get_text(" ", strip=True) if title_node else ""
    visible_text = _visible_text(soup)
    visible_words = _tokens(visible_text)
    all_tags = soup.find_all(True)
    tag_names = [tag.name.lower() for tag in all_tags if tag.name]
    scripts = soup.find_all("script")
    script_text = "\n".join(script.get_text(" ", strip=False) for script in scripts)

    anchors = soup.find_all("a")
    anchors_with_href = [anchor for anchor in anchors if anchor.has_attr("href")]
    anchor_hrefs = [str(anchor.get("href") or "").strip() for anchor in anchors_with_href]
    placeholder_anchors = [href for href in anchor_hrefs if href.lower() in _PLACEHOLDER_HREFS]
    internal_anchors = [
        href for href in anchor_hrefs if href.lower() not in _PLACEHOLDER_HREFS and not _is_external(href, final_url)
    ]
    external_anchors = [href for href in anchor_hrefs if _is_external(href, final_url)]

    forms = soup.find_all("form")
    inputs = soup.find_all("input")
    input_types = [str(node.get("type") or "text").lower() for node in inputs]
    hidden_inputs = [node for node in inputs if str(node.get("type") or "").lower() == "hidden"]
    password_inputs = [node for node in inputs if str(node.get("type") or "").lower() == "password"]
    form_actions = [str(form.get("action") or "").strip() for form in forms]
    null_actions = [action for action in form_actions if action.lower() in _PLACEHOLDER_HREFS]
    external_actions = [action for action in form_actions if _is_external(action, final_url)]
    mailto_actions = [action for action in form_actions if action.lower().startswith("mailto:")]
    password_forms = [form for form in forms if form.find("input", attrs={"type": re.compile("^password$", re.I)})]

    def form_action(form: Tag) -> str:
        return str(form.get("action") or "").strip()

    credential_form = any(
        _CREDENTIAL_RE.search(
            " ".join(
                (
                    form.get_text(" ", strip=True),
                    " ".join(str(value) for value in form.attrs.values()),
                    " ".join(str(value) for node in form.find_all(["input", "button"]) for value in node.attrs.values()),
                )
            )
        )
        for form in forms
    )

    image_urls = [str(node.get("src") or "").strip() for node in soup.find_all("img") if node.get("src")]
    script_urls = [str(node.get("src") or "").strip() for node in scripts if node.get("src")]
    stylesheet_urls = [
        str(node.get("href") or "").strip()
        for node in soup.find_all("link")
        if node.get("href") and "stylesheet" in [str(item).lower() for item in (node.get("rel") or [])]
    ]
    iframe_urls = [str(node.get("src") or "").strip() for node in soup.find_all("iframe") if node.get("src")]
    favicon_urls = [
        str(node.get("href") or "").strip()
        for node in soup.find_all("link")
        if node.get("href") and any("icon" in str(item).lower() for item in (node.get("rel") or []))
    ]
    resource_urls = image_urls + script_urls + stylesheet_urls + iframe_urls + favicon_urls
    external_resources = [resource for resource in resource_urls if _is_external(resource, final_url)]
    external_resource_domains = {
        urlsplit(urljoin(final_url, resource)).hostname
        for resource in external_resources
        if urlsplit(urljoin(final_url, resource)).hostname
    }

    hidden_element_present = any(
        tag.has_attr("hidden")
        or str(tag.get("aria-hidden") or "").lower() == "true"
        or bool(_HIDDEN_STYLE_RE.search(str(tag.get("style") or "")))
        for tag in all_tags
    )
    onmouseover_count = sum(1 for tag in all_tags if tag.has_attr("onmouseover"))
    right_click_disabled = bool(_RIGHT_CLICK_RE.search(script_text)) or any(
        tag.has_attr("oncontextmenu") for tag in all_tags
    )
    privacy_link_present = any(
        _PRIVACY_RE.search(f"{anchor.get_text(' ', strip=True)} {anchor.get('href') or ''}") for anchor in anchors
    )

    title_tokens = set(_tokens(title))
    domain_tokens = set(_tokens(domain_token))
    registered_tokens = set(_tokens(registered_domain))
    subdomain_tokens = set(_tokens(subdomain))
    url_title_tokens = set(_tokens(f"{hostname} {path}"))
    title_overlap = len(title_tokens & url_title_tokens) / max(len(title_tokens | url_title_tokens), 1)

    def ratio(part: int, total: int) -> float:
        return part / total if total else 0.0

    features.update(
        {
            "html.length": len(html),
            "html.visible_text_length": len(visible_text),
            "html.visible_word_count": len(visible_words),
            "html.visible_text_entropy": _entropy(visible_text),
            "html.visible_text_to_html_ratio": ratio(len(visible_text), len(html)),
            "html.total_tag_count": len(tag_names),
            "html.unique_tag_count": len(set(tag_names)),
            "html.div_count": tag_names.count("div"),
            "html.span_count": tag_names.count("span"),
            "html.paragraph_count": tag_names.count("p"),
            "html.heading_h1_h2_h3_count": sum(tag_names.count(name) for name in ("h1", "h2", "h3")),
            "html.list_count": sum(tag_names.count(name) for name in ("ul", "ol", "dl")),
            "html.table_count": tag_names.count("table"),
            "html.title_length": len(title),
            "html.navigation_present": int(bool(soup.find("nav") or soup.find(attrs={"role": "navigation"}))),
            "html.footer_present": int(bool(soup.find("footer") or soup.find(attrs={"role": "contentinfo"}))),
            "html.privacy_or_terms_link_present": int(privacy_link_present),
            "html.anchors_with_href_count": len(anchors_with_href),
            "html.anchors_missing_href_count": len(anchors) - len(anchors_with_href),
            "html.internal_anchor_count": len(internal_anchors),
            "html.external_anchor_count": len(external_anchors),
            "html.external_anchor_ratio": ratio(len(external_anchors), len(anchors_with_href)),
            "html.external_to_internal_anchor_ratio": len(external_anchors) / max(len(internal_anchors), 1),
            "html.null_or_empty_anchor_count": len(placeholder_anchors),
            "html.placeholder_link_ratio": ratio(len(placeholder_anchors), len(anchors_with_href)),
            "html.form_count": len(forms),
            "html.post_form_count": sum(str(form.get("method") or "").lower() == "post" for form in forms),
            "html.input_count": len(inputs),
            "html.text_input_count": sum(item in {"", "text", "search", "tel", "url"} for item in input_types),
            "html.password_input_count": len(password_inputs),
            "html.email_input_count": input_types.count("email"),
            "html.hidden_input_count": len(hidden_inputs),
            "html.hidden_input_ratio": ratio(len(hidden_inputs), len(inputs)),
            "html.submit_button_count": sum(item in {"submit", "image"} for item in input_types)
            + sum(str(button.get("type") or "submit").lower() == "submit" for button in soup.find_all("button")),
            "html.credential_form_present": int(credential_form),
            "html.null_form_action_count": len(null_actions),
            "html.external_form_action_count": len(external_actions),
            "html.mailto_form_action_count": len(mailto_actions),
            "html.password_form_null_action_present": int(
                any(form_action(form).lower() in _PLACEHOLDER_HREFS for form in password_forms)
            ),
            "html.password_form_external_action_present": int(
                any(_is_external(form_action(form), final_url) for form in password_forms)
            ),
            "html.image_count": len(image_urls),
            "html.external_image_count": sum(_is_external(url, final_url) for url in image_urls),
            "html.external_image_ratio": ratio(
                sum(_is_external(url, final_url) for url in image_urls), len(image_urls)
            ),
            "html.script_tag_count": len(scripts),
            "html.external_script_count": len(script_urls),
            "html.inline_script_count": len(scripts) - len(script_urls),
            "html.external_script_ratio": ratio(len(script_urls), len(scripts)),
            "html.stylesheet_link_count": len(stylesheet_urls),
            "html.external_stylesheet_count": sum(_is_external(url, final_url) for url in stylesheet_urls),
            "html.external_stylesheet_ratio": ratio(
                sum(_is_external(url, final_url) for url in stylesheet_urls), len(stylesheet_urls)
            ),
            "html.iframe_count": len(iframe_urls),
            "html.external_iframe_count": sum(_is_external(url, final_url) for url in iframe_urls),
            "html.favicon_count": len(favicon_urls),
            "html.external_favicon_count": sum(_is_external(url, final_url) for url in favicon_urls),
            "html.resource_url_count": len(resource_urls),
            "html.external_resource_ratio": ratio(len(external_resources), len(resource_urls)),
            "html.unique_external_resource_domain_count": len(external_resource_domains),
            "html.meta_tag_count": len(soup.find_all("meta")),
            "html.meta_refresh_count": sum(
                str(node.get("http-equiv") or "").lower() == "refresh" for node in soup.find_all("meta")
            ),
            "html.hidden_element_present": int(hidden_element_present),
            "html.javascript_redirect_present": int(bool(_JS_REDIRECT_RE.search(script_text))),
            "html.eval_call_count": len(re.findall(r"\beval\s*\(", script_text, re.IGNORECASE)),
            "html.atob_call_count": len(re.findall(r"\batob\s*\(", script_text, re.IGNORECASE)),
            "html.document_write_count": len(
                re.findall(r"\bdocument\s*\.\s*write(?:ln)?\s*\(", script_text, re.IGNORECASE)
            ),
            "html.right_click_disabling_present": int(right_click_disabled),
            "html.onmouseover_handler_count": onmouseover_count,
            "html.alert_or_popup_present": int(
                bool(re.search(r"\b(?:alert|confirm|prompt|window\.open)\s*\(", script_text, re.IGNORECASE))
            ),
            "html.current_domain_token_in_title": int(bool(title_tokens & domain_tokens)),
            "html.title_registered_domain_token_present": int(bool(title_tokens & registered_tokens)),
            "html.title_subdomain_token_present": int(bool(title_tokens & subdomain_tokens)),
            "html.title_url_token_overlap_ratio": title_overlap,
        }
    )

    history = metadata.get("redirect_history") or []
    redirect_urls = [requested_url] + [str(item.get("url") or "") for item in history if isinstance(item, dict)] + [
        final_url
    ]
    redirect_domains = [_registrable_domain(urlsplit(item).hostname or "") for item in redirect_urls if item]
    redirect_domain_changes = sum(
        left != right for left, right in zip(redirect_domains, redirect_domains[1:]) if left and right
    )
    features.update(
        {
            "metadata.redirect_count": int(metadata.get("redirect_count") or max(len(history), 0)),
            "metadata.final_url_changed": int(final_url.rstrip("/") != requested_url.rstrip("/")),
            "metadata.final_host_changed": int((requested.hostname or "").lower() != (final.hostname or "").lower()),
            "metadata.final_scheme_changed": int(requested.scheme.lower() != final.scheme.lower()),
            "metadata.redirect_domain_change_count": redirect_domain_changes,
        }
    )
    return features
