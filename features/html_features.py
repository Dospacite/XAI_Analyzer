#!/usr/bin/env python3
from __future__ import annotations

import re
from typing import Any

from . import common as c

JS_REDIRECT_RE = re.compile(
    r"(?:(?:window\s*\.\s*|document\s*\.\s*)?location(?:\s*\.\s*(?:href|replace|assign))?\s*=|(?:window\s*\.\s*)?location\s*\.\s*(?:replace|assign)\s*\()",
    re.I,
)


def input_type(tag: Any) -> str:
    return c.lowercase_attr(tag, "type")


def is_text_input(tag: Any) -> bool:
    return input_type(tag) in c.TEXT_INPUT_TYPES


def textish_attrs(tag: Any) -> str:
    return " ".join(
        value
        for value in [
            c.tag_attr(tag, "name"),
            c.tag_attr(tag, "id"),
            c.tag_attr(tag, "placeholder"),
            c.tag_attr(tag, "autocomplete"),
            c.tag_attr(tag, "aria-label"),
        ]
        if value
    )


def is_password_input(tag: Any) -> bool:
    return input_type(tag) == "password" or bool(re.search(r"\bpass(word)?\b", textish_attrs(tag), re.I))


def is_email_input(tag: Any) -> bool:
    return input_type(tag) == "email" or bool(re.search(r"\bemail\b", textish_attrs(tag), re.I))


def is_submit_button(tag: Any) -> bool:
    tag_name = getattr(getattr(tag, "root", None), "tag", "").lower()
    tag_type = input_type(tag)
    if tag_name == "button":
        return tag_type in {"", "submit"}
    return tag_name == "input" and tag_type == "submit"


def count(selector: Any, query: str) -> int:
    return len(c.css_select(selector, query))


def externality_counts(selector: Any, base_url: str, query: str, attr: str) -> tuple[int, int, int, int]:
    internal = external = null = total = 0
    for tag in c.css_select(selector, query):
        value = c.tag_attr(tag, attr)
        if value is None:
            null += 1
            continue
        total += 1
        if c.is_null_reference(value):
            null += 1
            continue
        kind = c.classify_link_target(base_url, value)
        if kind == "internal":
            internal += 1
        elif kind == "external":
            external += 1
    return internal, external, null, total


def hidden_element_indicator(selector: Any) -> int:
    for element in selector.xpath("//*"):
        if c.lowercase_attr(element, "hidden"):
            return 1
        if c.lowercase_attr(element, "aria-hidden") == "true":
            return 1
        if input_type(element) == "hidden":
            return 1
        style = c.tag_attr(element, "style") or ""
        style_l = re.sub(r"\s+", "", style.lower())
        if any(marker in style_l for marker in c.HIDDEN_STYLE_MARKERS):
            return 1
        if "width:0" in style_l and "height:0" in style_l:
            return 1
    return 0


def domain_token_in_title(title: str, final_url: str) -> int:
    domain_token = c.registered_domain_token(final_url)
    if not title or len(domain_token) < 3:
        return 0
    normalized_title = re.sub(r"[^a-z0-9]+", " ", title.lower())
    return 1 if domain_token in normalized_title.split() or domain_token in normalized_title else 0


def text_tokens(value: str) -> set[str]:
    return {token for token in re.split(r"[^a-z0-9]+", (value or "").lower()) if len(token) >= 2}


def title_url_overlap(title: str, final_url: str) -> tuple[float, int, int]:
    title_tokens = text_tokens(title)
    extracted = c.tld_parts(final_url)
    registered_token = extracted.domain.lower()
    url_token_source = " ".join(
        [
            extracted.subdomain,
            extracted.domain,
            c.parse_url(final_url).path,
            c.parse_url(final_url).query,
        ]
    )
    url_tokens = text_tokens(url_token_source)
    overlap = title_tokens & url_tokens
    overlap_ratio = len(overlap) / max(1, len(title_tokens))
    registered_present = 1 if registered_token and registered_token in title_tokens else 0
    subdomain_present = 1 if title_tokens & text_tokens(extracted.subdomain) else 0
    return overlap_ratio, registered_present, subdomain_present


def resolved_domain(base_url: str, reference: str | None) -> str | None:
    if not reference or c.is_null_reference(reference):
        return None
    if reference.lower().startswith(("javascript:", "mailto:", "tel:", "data:")):
        return None
    resolved = c.safe_urljoin(base_url, reference)
    domain = c.registrable_domain(resolved)
    return domain or None


def form_action_kind(base_url: str, action: str | None) -> str:
    if action is None or c.is_null_reference(action):
        return "null"
    lowered = action.strip().lower()
    if lowered.startswith("mailto:"):
        return "mailto"
    return c.classify_link_target(base_url, action) or "null"


def extract(normalized_doc: dict[str, Any]) -> dict[str, Any]:
    _requested_url, final_url = c.normalized_urls(normalized_doc)
    html = normalized_doc.get("html") or ""
    selector = c.html_selector(html)
    title = c.title_from_doc_or_html(normalized_doc, selector)
    text = c.visible_text(selector)
    title_overlap_ratio, title_registered_present, title_subdomain_present = title_url_overlap(title, final_url)

    anchors = c.css_select(selector, "a")
    anchors_with_href = sum(1 for tag in anchors if c.tag_attr(tag, "href") is not None)
    anchors_missing_href = max(0, len(anchors) - anchors_with_href)
    anchor_internal, anchor_external, anchor_null, _anchor_total = externality_counts(selector, final_url, "a", "href")

    form_tags = c.css_select(selector, "form")
    form_missing_action = 0
    form_null_action = 0
    form_external_action = 0
    form_mailto_action = 0
    form_post = 0
    credential_form_present = 0
    password_form_external_action_present = 0
    password_form_null_action_present = 0
    for form in form_tags:
        action = c.tag_attr(form, "action")
        method = c.lowercase_attr(form, "method")
        action_kind = form_action_kind(final_url, action)
        form_inputs = c.css_select(form, "input, textarea, select")
        form_has_password = any(is_password_input(tag) for tag in form_inputs)
        form_has_submit = any(is_submit_button(tag) for tag in c.css_select(form, "button, input"))
        if form_has_password and form_has_submit:
            credential_form_present = 1
        if form_has_password and action_kind == "external":
            password_form_external_action_present = 1
        if form_has_password and action_kind == "null":
            password_form_null_action_present = 1
        if method == "post":
            form_post += 1
        if action is None:
            form_missing_action += 1
            form_null_action += 1
            continue
        if action.lower().startswith("mailto:"):
            form_mailto_action += 1
        if c.is_null_reference(action):
            form_null_action += 1
        elif c.classify_link_target(final_url, action) == "external":
            form_external_action += 1

    input_tags = c.css_select(selector, "input")
    button_tags = c.css_select(selector, "button")
    iframe_internal, iframe_external, _iframe_null, _iframe_total = externality_counts(selector, final_url, "iframe[src]", "src")
    script_internal, script_external, _script_null, _script_total = externality_counts(selector, final_url, "script[src]", "src")
    style_internal, style_external, _style_null, _style_total = externality_counts(selector, final_url, "link[rel*=stylesheet][href]", "href")
    image_internal, image_external, _image_null, _image_total = externality_counts(selector, final_url, "img[src]", "src")
    favicon_internal, favicon_external, _favicon_null, _favicon_total = externality_counts(selector, final_url, "link[rel*=icon][href]", "href")

    resource_selectors = [
        ("script[src]", "src"),
        ("link[rel*=stylesheet][href]", "href"),
        ("img[src]", "src"),
        ("link[rel*=icon][href]", "href"),
        ("iframe[src]", "src"),
        ("audio[src],video[src],source[src]", "src"),
    ]
    resource_total = resource_external = 0
    unique_external_resource_domains: set[str] = set()
    for query, attr in resource_selectors:
        internal, external, _null, total = externality_counts(selector, final_url, query, attr)
        resource_total += total
        resource_external += external
        for tag in c.css_select(selector, query):
            value = c.tag_attr(tag, attr)
            if c.classify_link_target(final_url, value or "") == "external":
                domain = resolved_domain(final_url, value)
                if domain:
                    unique_external_resource_domains.add(domain)

    clickable_buttons = len(
        selector.xpath(
            "//button"
            " | //input[translate(@type,'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz')='submit'"
            "          or translate(@type,'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz')='button'"
            "          or translate(@type,'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz')='reset']"
            " | //*[@role='button' and not(self::button)]"
        )
    )

    return {
        "html.length": len(html),
        "html.visible_text_length": len(text),
        "html.visible_text_entropy": c.entropy(text),
        "html.visible_word_count": len(text.split()) if text else 0,
        "html.visible_text_to_html_ratio": len(text) / max(1, len(html)),
        "html.title_length": len(title),
        "html.current_domain_token_in_title": domain_token_in_title(title, final_url),
        "html.title_url_token_overlap_ratio": title_overlap_ratio,
        "html.title_registered_domain_token_present": title_registered_present,
        "html.title_subdomain_token_present": title_subdomain_present,
        "html.meta_tag_count": count(selector, "meta"),
        "html.meta_refresh_count": sum(1 for tag in c.css_select(selector, "meta") if c.lowercase_attr(tag, "http-equiv") == "refresh"),
        "html.total_tag_count": len(selector.xpath("//*")),
        "html.unique_tag_count": len({getattr(node.root, "tag", "").lower() for node in selector.xpath("//*")}),
        "html.anchors_with_href_count": anchors_with_href,
        "html.anchors_missing_href_count": anchors_missing_href,
        "html.null_or_empty_anchor_count": anchor_null,
        "html.placeholder_link_ratio": anchor_null / max(1, anchors_with_href + anchors_missing_href),
        "html.internal_anchor_count": anchor_internal,
        "html.external_anchor_count": anchor_external,
        "html.external_anchor_ratio": anchor_external / max(1, anchors_with_href),
        "html.external_to_internal_anchor_ratio": anchor_external / max(1, anchor_internal),
        "html.form_count": len(form_tags),
        "html.credential_form_present": credential_form_present,
        "html.password_form_external_action_present": password_form_external_action_present,
        "html.password_form_null_action_present": password_form_null_action_present,
        "html.null_form_action_count": form_null_action,
        "html.external_form_action_count": form_external_action,
        "html.mailto_form_action_count": form_mailto_action,
        "html.post_form_count": form_post,
        "html.input_count": len(input_tags),
        "html.text_input_count": sum(1 for tag in input_tags if is_text_input(tag)),
        "html.password_input_count": sum(1 for tag in input_tags if is_password_input(tag)),
        "html.email_input_count": sum(1 for tag in input_tags if is_email_input(tag)),
        "html.hidden_input_count": sum(1 for tag in input_tags if input_type(tag) == "hidden"),
        "html.hidden_input_ratio": sum(1 for tag in input_tags if input_type(tag) == "hidden") / max(1, len(input_tags)),
        "html.submit_button_count": sum(1 for tag in button_tags if is_submit_button(tag)) + sum(1 for tag in input_tags if is_submit_button(tag)),
        "html.iframe_count": count(selector, "iframe"),
        "html.external_iframe_count": iframe_external,
        "html.script_tag_count": count(selector, "script"),
        "html.external_script_count": script_external,
        "html.external_script_ratio": script_external / max(1, script_internal + script_external),
        "html.inline_script_count": count(selector, "script:not([src])"),
        "html.stylesheet_link_count": count(selector, "link[rel*=stylesheet][href]"),
        "html.external_stylesheet_count": style_external,
        "html.external_stylesheet_ratio": style_external / max(1, style_internal + style_external),
        "html.image_count": count(selector, "img"),
        "html.external_image_count": image_external,
        "html.external_image_ratio": image_external / max(1, image_internal + image_external),
        "html.favicon_count": count(selector, "link[rel*=icon][href]"),
        "html.external_favicon_count": favicon_external,
        "html.resource_url_count": resource_total,
        "html.unique_external_resource_domain_count": len(unique_external_resource_domains),
        "html.external_resource_ratio": resource_external / max(1, resource_total),
        "html.hidden_element_present": hidden_element_indicator(selector),
        "html.javascript_redirect_present": 1 if JS_REDIRECT_RE.search(html) else 0,
        "html.eval_call_count": len(re.findall(r"\beval\s*\(", html, flags=re.I)),
        "html.atob_call_count": len(re.findall(r"\batob\s*\(", html, flags=re.I)),
        "html.document_write_count": len(re.findall(r"\bdocument\s*\.\s*write(?:ln)?\s*\(", html, flags=re.I)),
        "html.alert_or_popup_present": 1 if re.search(r"\b(?:window\s*\.\s*)?(?:alert|open)\s*\(", html, flags=re.I) else 0,
        "html.onmouseover_handler_count": len(re.findall(r"\bonmouseover\b", html, flags=re.I)),
        "html.right_click_disabling_present": 1 if re.search(r"contextmenu|event\.button\s*={1,2}\s*2|preventdefault\s*\(", html, flags=re.I) else 0,
        "html.paragraph_count": count(selector, "p"),
        "html.div_count": count(selector, "div"),
        "html.span_count": count(selector, "span"),
        "html.table_count": count(selector, "table"),
        "html.list_count": len(selector.xpath("//ul | //ol | //dl")),
        "html.heading_h1_h2_h3_count": count(selector, "h1") + count(selector, "h2") + count(selector, "h3"),
        "html.footer_present": 1 if count(selector, "footer") else 0,
        "html.navigation_present": 1 if count(selector, "nav") else 0,
        "html.privacy_or_terms_link_present": 1
        if any(
            re.search(r"\b(?:privacy|terms|policy|conditions)\b", " ".join(filter(None, [c.selector_text(anchor, 120), c.tag_attr(anchor, "href") or ""])), flags=re.I)
            for anchor in anchors
        )
        else 0,
    }
