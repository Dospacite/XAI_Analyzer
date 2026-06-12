#!/usr/bin/env python3
"""Extract deterministic phishing evidence candidates from Stage A JSONL.

Input records are expected to come from extract_inputs_jsonl.py. This script is
Stage B only: it emits feature candidates, not final model verdicts.
"""

from __future__ import annotations

import argparse
from difflib import SequenceMatcher
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urljoin, urlparse

import tldextract
from tqdm import tqdm


extract_tld = tldextract.TLDExtract(suffix_list_urls=())

GENERIC_TITLES = {
    "sign in",
    "login",
    "log in",
    "secure portal",
    "document review",
    "account verification",
    "verification",
    "account",
    "portal",
    "home",
    "welcome",
    "index",
}

CREDENTIAL_TERMS = {
    "email",
    "e-mail",
    "password",
    "login",
    "log in",
    "sign in",
    "signin",
    "verify",
    "account",
    "security",
    "username",
    "user id",
}

URGENCY_TERMS = {
    "verify immediately",
    "suspended",
    "unusual activity",
    "confirm within",
    "limited",
    "locked",
    "expire",
    "expired",
    "security alert",
    "urgent",
    "immediately",
}

FILE_LURE_TERMS = {
    "invoice",
    "voicemail",
    "voice mail",
    "fax",
    "secure document",
    "shared file",
    "payment",
    "remittance",
    "review document",
    "document review",
    "shared document",
    "download file",
}

HELP_TERMS = {
    "forgot password",
    "reset password",
    "help",
    "support",
    "create account",
    "contact",
    "register",
    "sign up",
}

INTERSTITIAL_TERMS = {
    "checking your browser",
    "access denied",
    "captcha",
    "consent",
    "privacy choices",
    "privacy settings",
    "just a moment",
    "enable cookies",
}

BRAND_ALIASES: dict[str, set[str]] = {
    "microsoft": {"microsoft", "outlook", "office 365", "office365", "onedrive", "sharepoint", "webmail exchange"},
    "google": {"google", "gmail", "google drive", "workspace"},
    "apple": {"apple", "icloud", "apple id", "app store"},
    "paypal": {"paypal"},
    "amazon": {"amazon", "aws"},
    "meta": {"facebook", "meta", "instagram", "whatsapp"},
    "netflix": {"netflix"},
    "adobe": {"adobe", "acrobat"},
    "dropbox": {"dropbox"},
    "linkedin": {"linkedin"},
    "yahoo": {"yahoo"},
    "naver": {"naver"},
    "dhl": {"dhl"},
    "fedex": {"fedex"},
    "ups": {"ups"},
    "docuSign": {"docusign"},
}

BRAND_DOMAINS: dict[str, set[str]] = {
    "microsoft": {"microsoft.com", "microsoftonline.com", "live.com", "office.com", "office365.com", "outlook.com", "sharepoint.com", "onedrive.com"},
    "google": {"google.com", "gmail.com", "googleusercontent.com", "gstatic.com", "googleapis.com"},
    "apple": {"apple.com", "icloud.com", "mzstatic.com"},
    "paypal": {"paypal.com", "paypalobjects.com"},
    "amazon": {"amazon.com", "aws.amazon.com", "amazonaws.com"},
    "meta": {"facebook.com", "meta.com", "instagram.com", "whatsapp.com", "fbcdn.net"},
    "netflix": {"netflix.com"},
    "adobe": {"adobe.com", "adobe.io"},
    "dropbox": {"dropbox.com", "dropboxusercontent.com"},
    "linkedin": {"linkedin.com"},
    "yahoo": {"yahoo.com"},
    "naver": {"naver.com"},
    "dhl": {"dhl.com"},
    "fedex": {"fedex.com"},
    "ups": {"ups.com"},
    "docuSign": {"docusign.com", "docusign.net"},
}

KNOWN_IDP_DOMAINS = {
    "accounts.google.com",
    "login.microsoftonline.com",
    "microsoftonline.com",
    "okta.com",
    "auth0.com",
    "onelogin.com",
    "duosecurity.com",
    "pingidentity.com",
    "pingone.com",
    "login.salesforce.com",
    "salesforce.com",
    "workday.com",
    "myworkday.com",
}

KNOWN_INFRA_DOMAINS = {
    "cloudflare.com",
    "cloudfront.net",
    "akamaihd.net",
    "akamaiedge.net",
    "amazonaws.com",
    "azureedge.net",
    "github.io",
    "netlify.app",
    "vercel.app",
    "pages.dev",
    "web.app",
    "firebaseapp.com",
    "appspot.com",
    "herokuapp.com",
    "wixsite.com",
    "squarespace.com",
    "shopify.com",
    "myshopify.com",
}

KNOWN_PAYMENT_OR_APP_DOMAINS = {
    "stripe.com",
    "paypal.com",
    "checkout.com",
    "adyen.com",
    "squareup.com",
    "braintreepayments.com",
    "play.google.com",
    "apps.apple.com",
    "itunes.apple.com",
}

KNOWN_SOCIAL_OR_PROFILE_DOMAINS = {
    "facebook.com",
    "instagram.com",
    "linkedin.com",
    "twitter.com",
    "x.com",
    "youtube.com",
    "tiktok.com",
    "threads.net",
}

EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
CONFUSABLE_TRANSLATION = str.maketrans(
    {
        "0": "o",
        "1": "l",
        "3": "e",
        "4": "a",
        "5": "s",
        "7": "t",
        "8": "b",
        "9": "g",
        "@": "a",
        "$": "s",
    }
)

FEATURE_SUPERVISION: dict[str, dict[str, Any]] = {
    "url.long_url": {
        "type": "threshold_heuristic",
        "primary_eligible": False,
        "calibration_required": True,
        "note": "Store raw length and calibrate thresholds on the train split.",
    },
    "url.deep_subdomain": {
        "type": "threshold_heuristic",
        "primary_eligible": False,
        "calibration_required": True,
        "note": "Discount common SaaS and tenant hosts before using as evidence.",
    },
    "redirect.multi_hop": {
        "type": "threshold_heuristic",
        "primary_eligible": False,
        "calibration_required": True,
        "note": "Keep redirect count raw and weight by corpus and page type.",
    },
    "redirect.cross_domain": {
        "type": "relationship_heuristic",
        "primary_eligible": False,
        "calibration_required": False,
        "note": "Depends on a small domain relationship allowlist.",
    },
    "meta.robots_noindex_nofollow": {
        "type": "deterministic_weak_signal",
        "primary_eligible": False,
        "calibration_required": False,
        "note": "Negative-control risk; use as low-weight support only.",
    },
    "credential.credential_terms_near_form": {
        "type": "lexicon_heuristic",
        "primary_eligible": False,
        "calibration_required": False,
        "note": "Lexical credential-intent proxy; prefer as secondary evidence.",
    },
    "form.blank_action_js_submission_suspected": {
        "type": "proxy_for_missing_behavior_inspection",
        "primary_eligible": False,
        "calibration_required": False,
        "note": "Inferred from blank action and submit UI, not real JS/XHR inspection.",
    },
    "form.hidden_inputs": {
        "type": "deterministic_weak_signal",
        "primary_eligible": False,
        "calibration_required": False,
        "note": "Negative-control risk; never use as primary evidence.",
    },
    "form.external_action": {
        "type": "relationship_heuristic",
        "primary_eligible": False,
        "calibration_required": False,
        "note": "Strength depends on domain relationship quality; use as primary only after stronger org resolution.",
    },
    "form.action_same_org_domain": {
        "type": "relationship_heuristic",
        "primary_eligible": False,
        "calibration_required": False,
        "note": "Benign candidate depends on coarse organization resolution.",
    },
    "brand.domain_lookalike": {
        "type": "string_similarity_heuristic",
        "primary_eligible": False,
        "calibration_required": True,
        "note": "Candidate only; require corroborating signals before model target evidence.",
    },
    "brand.title_domain_match_strong": {
        "type": "manual_brand_mapping_heuristic",
        "primary_eligible": False,
        "calibration_required": False,
        "note": "Depends on limited brand alias and domain tables.",
    },
    "brand.title_domain_mismatch": {
        "type": "manual_brand_mapping_heuristic",
        "primary_eligible": False,
        "calibration_required": False,
        "note": "Use only when explicit brand claim provenance is clear.",
    },
    "brand.favicon_domain_mismatch": {
        "type": "partial_relationship_heuristic",
        "primary_eligible": False,
        "calibration_required": False,
        "note": "Does not verify favicon brand ownership; support only.",
    },
    "content.coercive_urgency_near_form": {
        "type": "lexicon_heuristic",
        "primary_eligible": False,
        "calibration_required": False,
        "note": "Keyword proxy; prone to paraphrase misses and benign warnings.",
    },
    "content.file_lure_terms": {
        "type": "lexicon_heuristic",
        "primary_eligible": False,
        "calibration_required": False,
        "note": "Candidate only; require credential or impersonation context.",
    },
    "content.non_credential_transactional_page": {
        "type": "absence_based_heuristic",
        "primary_eligible": False,
        "calibration_required": False,
        "note": "Benign/uncertain support derived from absence of detected credential flow.",
    },
    "content.copyright_domain_mismatch": {
        "type": "regex_and_alias_heuristic",
        "primary_eligible": False,
        "calibration_required": False,
        "note": "Weak org parsing; support only until footer/legal parsing improves.",
    },
    "contact.identity_domain_mismatch": {
        "type": "narrow_relationship_proxy",
        "primary_eligible": False,
        "calibration_required": False,
        "note": "Currently focuses on visible emails and simple domain relationship.",
    },
    "support.contact_domain_match": {
        "type": "narrow_relationship_proxy",
        "primary_eligible": False,
        "calibration_required": False,
        "note": "Support/contact proxy from emails and anchor text.",
    },
    "link.null_or_void_anchors": {
        "type": "deterministic_weak_signal",
        "primary_eligible": False,
        "calibration_required": False,
        "note": "Negative-control risk for SPAs and menu UI.",
    },
    "link.brand_text_domain_mismatch": {
        "type": "alias_relationship_heuristic",
        "primary_eligible": False,
        "calibration_required": False,
        "note": "Depends on alias mapping and incomplete destination exceptions.",
    },
    "link.high_external_anchor_ratio": {
        "type": "threshold_heuristic",
        "primary_eligible": False,
        "calibration_required": True,
        "note": "Store counts and ratio; label after page-type classification.",
    },
    "navigation.functional_internal_links": {
        "type": "loose_counting_heuristic",
        "primary_eligible": False,
        "calibration_required": False,
        "note": "Proxy for navigation quality; should be tightened to nav-like anchors.",
    },
    "identity.provider_expected_domain": {
        "type": "allowlist_heuristic",
        "primary_eligible": False,
        "calibration_required": False,
        "note": "Short public IdP allowlist; enterprise SSO coverage is limited.",
    },
    "iframe.hidden_iframe": {
        "type": "deterministic_context_weak_signal",
        "primary_eligible": False,
        "calibration_required": False,
        "note": "Context weak; benign embeds and tracking can be hidden too.",
    },
    "login.missing_recovery_or_help_flow": {
        "type": "absence_based_lexical_heuristic",
        "primary_eligible": False,
        "calibration_required": False,
        "note": "Minimal legitimate login pages can lack recovery/help links.",
    },
    "page.low_semantic_content": {
        "type": "threshold_heuristic",
        "primary_eligible": False,
        "calibration_required": True,
        "note": "Uncertainty support; current threshold is intentionally crude.",
    },
    "page.generic_login_without_brand_claim": {
        "type": "alias_absence_heuristic",
        "primary_eligible": False,
        "calibration_required": False,
        "note": "Depends on limited brand and org-claim extraction.",
    },
    "page.rendering_incomplete_or_script_dependent": {
        "type": "proxy_heuristic",
        "primary_eligible": False,
        "calibration_required": True,
        "note": "Uses short text plus many scripts as a proxy for incomplete rendering.",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract deterministic feature candidates from Stage A JSONL."
    )
    parser.add_argument("--input", type=Path, required=True, help="Stage A input JSONL.")
    parser.add_argument("--output", type=Path, default=None, help="Output JSONL. Defaults to stdout.")
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bar.")
    parser.add_argument(
        "--max-features-per-record",
        type=int,
        default=0,
        help="Maximum features per record. 0 means no limit.",
    )
    parser.add_argument(
        "--long-url-p95",
        type=int,
        default=120,
        help="Initial long-URL threshold; tune on train split distribution.",
    )
    parser.add_argument(
        "--deep-subdomain-labels",
        type=int,
        default=4,
        help="Hostname label count threshold for deep-subdomain signal.",
    )
    parser.add_argument(
        "--multi-hop-redirects",
        type=int,
        default=2,
        help="Redirect count threshold for multi-hop signal.",
    )
    parser.add_argument(
        "--high-external-anchor-ratio",
        type=float,
        default=0.8,
        help="External anchor ratio threshold for weak external-link signal.",
    )
    return parser.parse_args()


def norm_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def lower_text(value: Any) -> str:
    return norm_text(value).lower()


def registrable_domain(url_or_host: str) -> str:
    raw = norm_text(url_or_host)
    if not raw:
        return ""
    parsed = urlparse(raw)
    host = parsed.hostname
    if host is None and "://" not in raw and "/" not in raw:
        host = raw
    if not host:
        return ""
    ext = extract_tld(host)
    if ext.domain and ext.suffix:
        return f"{ext.domain}.{ext.suffix}".lower()
    return host.lower()


def hostname(url: str) -> str:
    return (urlparse(norm_text(url)).hostname or "").lower()


def domain_in_set(domain: str, known_domains: set[str]) -> bool:
    if not domain:
        return False
    return any(domain == item or domain.endswith("." + item) for item in known_domains)


def relationship(source_domain: str, target_domain: str) -> str:
    source_raw = (source_domain or "").lower()
    target_raw = (target_domain or "").lower()
    if not source_raw or not target_raw:
        return "unknown"
    source_registrable = registrable_domain(source_raw)
    target_registrable = registrable_domain(target_raw)
    if source_registrable and target_registrable and source_registrable == target_registrable:
        return "same_registrable_domain"
    if domain_in_set(target_raw, KNOWN_PAYMENT_OR_APP_DOMAINS) or domain_in_set(
        target_registrable, KNOWN_PAYMENT_OR_APP_DOMAINS
    ):
        return "known_payment_or_app_store"
    if domain_in_set(target_raw, KNOWN_IDP_DOMAINS) or domain_in_set(target_registrable, KNOWN_IDP_DOMAINS):
        return "known_identity_provider"
    if domain_in_set(target_raw, KNOWN_INFRA_DOMAINS) or domain_in_set(target_registrable, KNOWN_INFRA_DOMAINS):
        return "known_infrastructure_provider"
    if same_known_brand(source_registrable or source_raw, target_registrable or target_raw):
        return "same_organization_or_alias"
    return "unrelated_third_party"


def known_provider_relationship(domain: str) -> str | None:
    domain = (domain or "").lower()
    registrable = registrable_domain(domain)
    if domain_in_set(domain, KNOWN_PAYMENT_OR_APP_DOMAINS) or domain_in_set(registrable, KNOWN_PAYMENT_OR_APP_DOMAINS):
        return "known_payment_or_app_store"
    if domain_in_set(domain, KNOWN_IDP_DOMAINS) or domain_in_set(registrable, KNOWN_IDP_DOMAINS):
        return "known_identity_provider"
    if domain_in_set(domain, KNOWN_INFRA_DOMAINS) or domain_in_set(registrable, KNOWN_INFRA_DOMAINS):
        return "known_infrastructure_provider"
    if domain_in_set(domain, KNOWN_SOCIAL_OR_PROFILE_DOMAINS) or domain_in_set(registrable, KNOWN_SOCIAL_OR_PROFILE_DOMAINS):
        return "known_social_or_profile_provider"
    return None


def same_known_brand(domain_a: str, domain_b: str) -> bool:
    for domains in BRAND_DOMAINS.values():
        if domain_in_set(domain_a, domains) and domain_in_set(domain_b, domains):
            return True
    return False


def brand_for_domain(domain: str) -> str | None:
    for brand, domains in BRAND_DOMAINS.items():
        if domain_in_set(domain, domains):
            return brand
    return None


def page_domain(page: dict[str, Any]) -> str:
    return registrable_domain(page.get("final_url") or page.get("url") or "")


def all_form_items(page: dict[str, Any]) -> list[dict[str, Any]]:
    forms = page.get("forms") or {}
    if isinstance(forms, dict):
        items = forms.get("items") or []
        return [item for item in items if isinstance(item, dict)]
    if isinstance(forms, list):
        return [item for item in forms if isinstance(item, dict)]
    return []


def all_anchor_items(page: dict[str, Any]) -> list[dict[str, Any]]:
    anchors = page.get("anchors") or {}
    if isinstance(anchors, dict):
        items = anchors.get("items") or []
        return [item for item in items if isinstance(item, dict)]
    if isinstance(anchors, list):
        return [item for item in anchors if isinstance(item, dict)]
    return []


def all_iframe_items(page: dict[str, Any]) -> list[dict[str, Any]]:
    iframes = page.get("iframes") or {}
    if isinstance(iframes, dict):
        items = iframes.get("items") or []
        return [item for item in items if isinstance(item, dict)]
    if isinstance(iframes, list):
        return [item for item in iframes if isinstance(item, dict)]
    return []


def all_inputs(form: dict[str, Any]) -> list[dict[str, Any]]:
    inputs = form.get("inputs") or []
    return [item for item in inputs if isinstance(item, dict)]


def form_text(form: dict[str, Any]) -> str:
    parts: list[str] = [norm_text(form.get("text"))]
    for input_item in all_inputs(form):
        parts.extend(
            [
                norm_text(input_item.get("label")),
                norm_text(input_item.get("placeholder")),
                norm_text(input_item.get("name")),
                norm_text(input_item.get("id")),
                norm_text(input_item.get("aria_label")),
            ]
        )
    for button in form.get("buttons") or []:
        parts.append(norm_text(button))
    return norm_text(" ".join(part for part in parts if part))


def anchor_text(anchor: dict[str, Any]) -> str:
    return norm_text(
        " ".join(
            [
                norm_text(anchor.get("text")),
                norm_text(anchor.get("title")),
                norm_text(anchor.get("aria_label")),
            ]
        )
    )


def is_meaningful_nav_anchor(anchor: dict[str, Any], base_url: str, base_domain: str) -> bool:
    href = norm_text(anchor.get("href"))
    href_l = href.lower()
    text = anchor_text(anchor)
    if not href or href_l == "#" or href_l.startswith(("javascript:", "mailto:", "tel:")):
        return False
    if len(text) < 2 or not re.search(r"[a-z0-9]", text, re.I):
        return False
    parsed = urlparse(href)
    href_domain = absolute_domain(base_url, href)
    rel = relationship(base_domain, href_domain)
    return rel in {"same_registrable_domain", "same_organization_or_alias"} or (
        not parsed.scheme and not href.startswith("#")
    )


def has_password_form(form: dict[str, Any]) -> bool:
    return any(lower_text(item.get("type")) == "password" for item in all_inputs(form))


def has_credential_form(page: dict[str, Any]) -> bool:
    return any(has_password_form(form) or matched_terms(form_text(form), CREDENTIAL_TERMS) for form in all_form_items(page))


def matched_terms(text: str, terms: set[str]) -> list[str]:
    text_l = lower_text(text)
    found = []
    for term in sorted(terms):
        if re.search(r"\b" + re.escape(term).replace(r"\ ", r"\s+") + r"\b", text_l):
            found.append(term)
    return found


def redact_email(value: str) -> str:
    def repl(match: re.Match[str]) -> str:
        email = match.group(0)
        _, _, domain = email.partition("@")
        return f"<email>@{domain}" if domain else "<email>"

    return EMAIL_RE.sub(repl, value)


def absolute_domain(base_url: str, maybe_url: str) -> str:
    if not maybe_url:
        return ""
    return hostname(urljoin(base_url, maybe_url))


def add_feature(
    features: list[dict[str, Any]],
    feature_id: str,
    direction: str,
    severity: str,
    value: dict[str, Any],
    statement: str,
) -> None:
    feature = {
        "id": feature_id,
        "direction": direction,
        "severity": severity,
        "value": value,
        "statement": statement,
    }
    supervision = FEATURE_SUPERVISION.get(feature_id)
    if supervision:
        feature["supervision"] = supervision
    features.append(feature)


def has_feature(features: list[dict[str, Any]], feature_id: str) -> bool:
    return any(feature.get("id") == feature_id for feature in features)


def extract_brand_claims(text: str) -> list[dict[str, str]]:
    text_l = lower_text(text)
    claims: list[dict[str, str]] = []
    if text_l in GENERIC_TITLES:
        return claims
    for canonical, aliases in BRAND_ALIASES.items():
        for alias in sorted(aliases, key=len, reverse=True):
            pattern = r"\b" + re.escape(alias).replace(r"\ ", r"\s+") + r"\b"
            if re.search(pattern, text_l):
                claims.append({"brand": canonical, "matched": alias})
                break
    return claims


def brand_matches_domain(brand: str, domain: str) -> bool:
    return domain_in_set(domain, BRAND_DOMAINS.get(brand, set()))


def decode_label(label: str) -> str:
    try:
        return label.encode("ascii").decode("idna")
    except UnicodeError:
        return label


def skeletonize_label(value: str) -> str:
    value = decode_label(lower_text(value))
    value = value.translate(CONFUSABLE_TRANSLATION)
    value = re.sub(r"[^a-z0-9]+", "", value)
    # Common visual substitutions in phishing domains. Keep this small and
    # explicit to avoid turning unrelated words into brand matches.
    value = value.replace("rn", "m").replace("vv", "w")
    return value


def brand_alias_skeletons() -> list[tuple[str, str, str]]:
    items: list[tuple[str, str, str]] = []
    for brand, aliases in BRAND_ALIASES.items():
        for alias in aliases:
            skeleton = skeletonize_label(alias)
            if len(skeleton) >= 5:
                items.append((brand, alias, skeleton))
    return items


BRAND_ALIAS_SKELETONS = brand_alias_skeletons()


def domain_lookalike(page_domain_value: str) -> dict[str, Any] | None:
    raw = lower_text(page_domain_value)
    host = hostname(raw)
    if not host and "://" not in raw:
        host = raw.split("/", 1)[0]
    domain = registrable_domain(host or raw)
    if not domain:
        return None
    if any(brand_matches_domain(brand, domain) for brand in BRAND_DOMAINS) or domain_in_set(domain, KNOWN_IDP_DOMAINS):
        return None

    extracted = extract_tld(host or domain)
    candidate_label = extracted.domain or domain.split(".", 1)[0]
    generic_labels = {"www", "login", "auth", "account", "accounts", "secure", "mail", "app", "cdn", "static"}
    host_labels = [label for label in (host or domain).split(".") if label and label not in generic_labels]
    raw_parts = [candidate_label] + host_labels
    for label in list(raw_parts):
        raw_parts.extend(re.split(r"[-_]+", label))
    label_skeletons = []
    for part in raw_parts:
        skeleton = skeletonize_label(part)
        if len(skeleton) >= 5 and skeleton not in label_skeletons:
            label_skeletons.append(skeleton)

    for label_skeleton in label_skeletons:
        for brand, alias, alias_skeleton in BRAND_ALIAS_SKELETONS:
            if brand_matches_domain(brand, domain):
                continue
            if alias_skeleton == label_skeleton:
                return {
                    "domain": domain,
                    "hostname": host or domain,
                    "target_brand": brand,
                    "matched_alias": alias,
                    "matched_label": label_skeleton,
                    "match_type": "confusable_exact",
                }
            if alias_skeleton in label_skeleton and len(alias_skeleton) >= 6:
                return {
                    "domain": domain,
                    "hostname": host or domain,
                    "target_brand": brand,
                    "matched_alias": alias,
                    "matched_label": label_skeleton,
                    "match_type": "brand_embedded_in_domain",
                }

            length_delta = abs(len(label_skeleton) - len(alias_skeleton))
            if len(alias_skeleton) >= 7 and length_delta <= 2:
                similarity = SequenceMatcher(None, label_skeleton, alias_skeleton).ratio()
                if similarity >= 0.88:
                    return {
                        "domain": domain,
                        "hostname": host or domain,
                        "target_brand": brand,
                        "matched_alias": alias,
                        "matched_label": label_skeleton,
                        "match_type": "near_edit_distance",
                        "similarity": round(similarity, 4),
                    }

    return None


def page_identity_text(page: dict[str, Any]) -> str:
    return norm_text(" ".join([norm_text(page.get("title")), norm_text(page.get("visible_text"))[:2000]]))


def extract_measurements(page: dict[str, Any]) -> dict[str, Any]:
    url = norm_text(page.get("url"))
    final_url = norm_text(page.get("final_url") or url)
    final_host = hostname(final_url)
    anchors = all_anchor_items(page)
    forms = all_form_items(page)
    resources = page.get("resources") or {}
    base_domain = page_domain(page)
    base_url = final_url or url

    external_anchor_count = 0
    internal_nav_candidate_count = 0
    null_anchor_count = 0
    contact_anchor_domains: set[str] = set()
    for anchor in anchors:
        href = norm_text(anchor.get("href"))
        href_l = href.lower()
        if not href or href_l == "#" or href_l.startswith("javascript:void"):
            null_anchor_count += 1
            continue
        href_domain = absolute_domain(base_url, href)
        rel = relationship(base_domain, href_domain)
        if href_domain and rel not in {"same_registrable_domain", "same_organization_or_alias"}:
            external_anchor_count += 1
        if is_meaningful_nav_anchor(anchor, base_url, base_domain):
            internal_nav_candidate_count += 1
        if any(term in lower_text(anchor_text(anchor)) for term in {"contact", "support", "help", "privacy", "legal"}):
            if href_domain:
                contact_anchor_domains.add(href_domain)

    password_input_count = 0
    hidden_input_count = 0
    credential_form_count = 0
    form_action_relationships: dict[str, int] = {}
    for form in forms:
        form_terms = matched_terms(form_text(form), CREDENTIAL_TERMS)
        is_credential = has_password_form(form) or bool(form_terms)
        if is_credential:
            credential_form_count += 1
        for input_item in all_inputs(form):
            input_type = lower_text(input_item.get("type"))
            if input_type == "password":
                password_input_count += 1
            elif input_type == "hidden" and is_credential:
                hidden_input_count += 1
        action = norm_text(form.get("action"))
        if is_credential and action:
            action_domain = absolute_domain(base_url, action)
            rel = relationship(base_domain, action_domain)
            form_action_relationships[rel] = form_action_relationships.get(rel, 0) + 1

    total_anchors = len(anchors)
    return {
        "url_length": len(url),
        "final_url_length": len(final_url),
        "final_hostname": final_host,
        "hostname_label_count": len(final_host.split(".")) if final_host else 0,
        "redirect_count": len(page.get("redirects") or []),
        "form_count": len(forms),
        "credential_form_count": credential_form_count,
        "password_input_count": password_input_count,
        "hidden_input_count": hidden_input_count,
        "form_action_relationship_counts": form_action_relationships,
        "anchor_count": total_anchors,
        "external_anchor_count": external_anchor_count,
        "external_anchor_ratio": round(external_anchor_count / total_anchors, 4) if total_anchors else 0.0,
        "null_anchor_count": null_anchor_count,
        "internal_nav_candidate_count": internal_nav_candidate_count,
        "contact_anchor_domains": sorted(contact_anchor_domains),
        "favicon_count": len(resources.get("favicon_hrefs") or []),
        "script_src_sample_count": len(resources.get("script_src_sample") or []),
        "stylesheet_href_sample_count": len(resources.get("stylesheet_href_sample") or []),
        "image_src_sample_count": len(resources.get("image_src_sample") or []),
        "visible_text_length": len(lower_text(page.get("visible_text"))),
        "title_is_generic": lower_text(page.get("title")) in GENERIC_TITLES,
    }


def extract_url_features(page: dict[str, Any], args: argparse.Namespace, features: list[dict[str, Any]]) -> None:
    url = norm_text(page.get("url"))
    final_url = norm_text(page.get("final_url") or url)
    final_host = hostname(final_url)
    final_domain = page_domain(page)
    label_count = len(final_host.split(".")) if final_host else 0

    if len(url) >= args.long_url_p95:
        add_feature(
            features,
            "url.long_url",
            "suspicious",
            "low",
            {"length": len(url), "threshold": args.long_url_p95, "threshold_source": "cli_uncalibrated"},
            f"URL is unusually long at {len(url)} characters.",
        )

    if label_count >= args.deep_subdomain_labels:
        provider_rel = known_provider_relationship(final_domain)
        add_feature(
            features,
            "url.deep_subdomain",
            "neutral" if provider_rel else "suspicious",
            "low",
            {
                "label_count": label_count,
                "threshold": args.deep_subdomain_labels,
                "threshold_source": "cli_uncalibrated",
                "hostname": final_host,
                "domain": final_domain,
                "provider_relationship": provider_rel,
            },
            f"Host has {label_count} dot-separated labels.",
        )

    redacted = redact_email(url)
    if redacted != url:
        add_feature(
            features,
            "url.email_identifier",
            "suspicious",
            "medium",
            {"url_sample": redacted[:300]},
            "URL contains an email-like identifier.",
        )

    if any(part.startswith("xn--") for part in final_host.split(".")) or any(ord(ch) > 127 for ch in final_host):
        add_feature(
            features,
            "url.homograph_or_unicode_hostname",
            "suspicious",
            "medium",
            {"hostname": final_host},
            "Hostname contains punycode or non-ASCII characters.",
        )

    if urlparse(final_url).scheme == "http" and has_credential_form(page):
        add_feature(
            features,
            "url.http_not_https",
            "suspicious",
            "medium",
            {"scheme": "http"},
            "Final page is served over HTTP rather than HTTPS.",
        )


def extract_redirect_features(page: dict[str, Any], args: argparse.Namespace, features: list[dict[str, Any]]) -> None:
    redirects = page.get("redirects") or []
    redirect_count = len(redirects)
    start_domain = registrable_domain(page.get("url") or "")
    final_domain = page_domain(page)

    if redirect_count >= args.multi_hop_redirects:
        redirect_domains = [
            registrable_domain(redirect.get("url") or "")
            for redirect in redirects
            if isinstance(redirect, dict) and redirect.get("url")
        ]
        add_feature(
            features,
            "redirect.multi_hop",
            "suspicious",
            "low",
            {
                "redirect_count": redirect_count,
                "redirect_domains": [domain for domain in redirect_domains if domain],
                "threshold": args.multi_hop_redirects,
                "threshold_source": "cli_uncalibrated",
            },
            f"Page reaches the final URL after {redirect_count} redirects.",
        )

    rel = relationship(start_domain, final_domain)
    if start_domain and final_domain and start_domain != final_domain:
        severity = "medium" if rel in {"unrelated_third_party", "unknown"} else "low"
        direction = "suspicious" if severity == "medium" else "neutral"
        add_feature(
            features,
            "redirect.cross_domain",
            direction,
            severity,
            {"start_domain": start_domain, "final_domain": final_domain, "relationship": rel},
            f"Redirect chain changes registrable domain from {start_domain} to {final_domain}.",
        )


def extract_meta_features(page: dict[str, Any], features: list[dict[str, Any]]) -> None:
    for meta in page.get("meta") or []:
        if not isinstance(meta, dict):
            continue
        name = lower_text(meta.get("name"))
        content = norm_text(meta.get("content"))
        if name == "robots" and re.search(r"\b(noindex|nofollow)\b", content, re.I):
            add_feature(
                features,
                "meta.robots_noindex_nofollow",
                "suspicious",
                "low",
                {"content": content},
                f"Page sets robots directive '{content}'.",
            )
            return


def extract_form_features(page: dict[str, Any], features: list[dict[str, Any]]) -> None:
    forms = all_form_items(page)
    page_base = norm_text(page.get("final_url") or page.get("url"))
    base_domain = page_domain(page)
    password_count = 0
    credential_terms: set[str] = set()
    empty_actions = 0
    hidden_names: list[str] = []
    external_actions: list[dict[str, str]] = []
    same_org_actions = 0
    js_only_candidates = 0

    for form in forms:
        text = form_text(form)
        form_terms = matched_terms(text, CREDENTIAL_TERMS)
        credential_terms.update(form_terms)
        is_credential = has_password_form(form) or bool(form_terms)
        action = norm_text(form.get("action"))

        for input_item in all_inputs(form):
            input_type = lower_text(input_item.get("type"))
            if input_type == "password":
                password_count += 1
            if input_type == "hidden" and is_credential:
                name = norm_text(input_item.get("name") or input_item.get("id"))
                if name:
                    hidden_names.append(name)

        if is_credential and (not action or action.lower() == "about:blank"):
            empty_actions += 1
            if form.get("buttons"):
                js_only_candidates += 1
        elif is_credential and action:
            action_domain = absolute_domain(page_base, action)
            rel = relationship(base_domain, action_domain)
            if rel in {"same_registrable_domain", "same_organization_or_alias"}:
                same_org_actions += 1
            elif rel == "known_identity_provider":
                add_feature(
                    features,
                    "identity.provider_expected_domain",
                    "benign",
                    "medium",
                    {"provider_domain": action_domain, "relationship": rel},
                    f"Authentication uses an expected identity provider domain: {action_domain}.",
                )
            elif action_domain:
                external_actions.append({"action_domain": action_domain, "relationship": rel})

    if password_count:
        add_feature(
            features,
            "credential.password_input_present",
            "suspicious",
            "high",
            {"count": password_count},
            f"Page contains {password_count} password field(s).",
        )

    if credential_terms:
        terms = sorted(credential_terms)
        add_feature(
            features,
            "credential.credential_terms_near_form",
            "suspicious",
            "medium",
            {"terms": terms},
            "Form asks for credential-related information: " + ", ".join(terms) + ".",
        )

    if empty_actions:
        add_feature(
            features,
            "form.empty_or_blank_action",
            "suspicious",
            "medium",
            {"count": empty_actions},
            "Form uses an empty or blank action.",
        )

    if js_only_candidates:
        add_feature(
            features,
            "form.blank_action_js_submission_suspected",
            "uncertain",
            "low",
            {"count": js_only_candidates},
            "Form has a blank action and may rely on JavaScript submission.",
        )

    if hidden_names:
        unique_names = sorted(set(hidden_names))[:10]
        add_feature(
            features,
            "form.hidden_inputs",
            "suspicious",
            "low",
            {"names": unique_names, "count": len(hidden_names)},
            "Credential form includes hidden input field(s): " + ", ".join(unique_names) + ".",
        )

    for item in external_actions[:5]:
        action_domain = item["action_domain"]
        rel = item["relationship"]
        suspicious_external = rel in {"unrelated_third_party", "unknown"}
        add_feature(
            features,
            "form.external_action",
            "suspicious" if suspicious_external else "neutral",
            "medium" if suspicious_external else "low",
            {"action_domain": action_domain, "page_domain": base_domain, "relationship": rel},
            f"Form submits to {action_domain}, which differs from page domain {base_domain}.",
        )

    if same_org_actions:
        add_feature(
            features,
            "form.action_same_org_domain",
            "benign",
            "low",
            {"count": same_org_actions, "relationship": "same_registrable_domain_or_known_alias"},
            "Form submission stays within the same organization domain.",
        )


def extract_brand_features(page: dict[str, Any], features: list[dict[str, Any]]) -> None:
    domain = page_domain(page)
    title = norm_text(page.get("title"))
    title_claims = extract_brand_claims(title)
    page_claims = title_claims or extract_brand_claims(page_identity_text(page))
    claim_provenance = "title" if title_claims else "title_or_visible_text"
    lookalike = domain_lookalike(page.get("final_url") or page.get("url") or domain)

    if lookalike:
        add_feature(
            features,
            "brand.domain_lookalike",
            "suspicious",
            "medium",
            lookalike,
            f"Page hostname {lookalike['hostname']} looks like {lookalike['target_brand']} but is not a known {lookalike['target_brand']} domain.",
        )

    for claim in page_claims[:3]:
        brand = claim["brand"]
        if brand_matches_domain(brand, domain):
            add_feature(
                features,
                "brand.title_domain_match_strong",
                "benign",
                "medium",
                {
                    "domain": domain,
                    "claimed_brand": brand,
                    "matched": claim["matched"],
                    "claim_provenance": claim_provenance,
                },
                f"Page identity is consistent with the registrable domain {domain}.",
            )
        elif title_claims:
            add_feature(
                features,
                "brand.title_domain_mismatch",
                "suspicious",
                "medium",
                {
                    "title": title,
                    "domain": domain,
                    "claimed_brand": brand,
                    "matched": claim["matched"],
                    "claim_provenance": "title",
                    "relationship": "not_known_brand_domain",
                    "claimed_brand_known_domains": sorted(BRAND_DOMAINS.get(brand, []))[:5],
                },
                f"Page title claims '{title}', but the registrable domain is {domain}.",
            )

    resources = page.get("resources") or {}
    for href in resources.get("favicon_hrefs") or []:
        favicon_domain = absolute_domain(norm_text(page.get("final_url") or page.get("url")), norm_text(href))
        rel = relationship(domain, favicon_domain)
        favicon_brand = brand_for_domain(favicon_domain)
        page_brand = brand_for_domain(domain)
        if favicon_domain and favicon_domain != domain and favicon_brand and favicon_brand != page_brand:
            add_feature(
                features,
                "brand.favicon_domain_mismatch",
                "suspicious",
                "low",
                {
                    "favicon_domain": favicon_domain,
                    "favicon_brand": favicon_brand,
                    "page_domain": domain,
                    "page_brand": page_brand,
                    "relationship": rel,
                },
                f"Favicon is loaded from {favicon_domain}, which differs from the page domain {domain}.",
            )
            break


def extract_content_features(page: dict[str, Any], features: list[dict[str, Any]]) -> None:
    domain = page_domain(page)
    text = page_identity_text(page)
    form_texts = " ".join(form_text(form) for form in all_form_items(page))

    urgency = matched_terms(form_texts, URGENCY_TERMS)
    if urgency:
        add_feature(
            features,
            "content.coercive_urgency_near_form",
            "suspicious",
            "medium",
            {"terms": urgency},
            "Form-adjacent text uses urgent or coercive wording: " + ", ".join(urgency) + ".",
        )

    lures = matched_terms(text, FILE_LURE_TERMS)
    if lures:
        feature_id = "content.file_lure_terms"
        direction = "suspicious" if has_credential_form(page) else "benign"
        statement = "Page uses file or transaction lure wording: " + ", ".join(lures) + "."
        if not has_credential_form(page):
            feature_id = "content.non_credential_transactional_page"
            direction = "uncertain"
            statement = "Page uses transactional wording but does not request credentials."
        severity = "medium" if feature_id == "content.file_lure_terms" else "low"
        add_feature(features, feature_id, direction, severity, {"terms": lures}, statement)

    copyright_match = re.search(
        r"(?:copyright|\(c\)|©|all rights reserved|trademark|privacy policy)\s+(.{0,120})",
        text,
        re.I,
    )
    if copyright_match:
        snippet = norm_text(copyright_match.group(0))[:160]
        claims = extract_brand_claims(snippet)
        for claim in claims:
            if not brand_matches_domain(claim["brand"], domain):
                add_feature(
                    features,
                    "content.copyright_domain_mismatch",
                    "suspicious",
                    "medium",
                    {"claimed_owner": claim["matched"], "domain": domain, "snippet": snippet},
                    f"Footer or legal text names '{claim['matched']}', which does not match {domain}.",
                )
                break

    contact_records: list[dict[str, str]] = []
    emails = EMAIL_RE.findall(text)
    for email in emails[:10]:
        email_domain = registrable_domain(email.split("@", 1)[1])
        if email_domain:
            contact_records.append(
                {
                    "kind": "email",
                    "contact": redact_email(email),
                    "contact_domain": email_domain,
                    "relationship": relationship(domain, email_domain),
                }
            )

    page_base = norm_text(page.get("final_url") or page.get("url"))
    for anchor in all_anchor_items(page):
        text_l = lower_text(anchor_text(anchor))
        if not any(term in text_l for term in {"contact", "support", "help", "privacy", "legal"}):
            continue
        href_domain = absolute_domain(page_base, norm_text(anchor.get("href")))
        if href_domain:
            contact_records.append(
                {
                    "kind": "link",
                    "contact": href_domain,
                    "contact_domain": href_domain,
                    "relationship": relationship(domain, href_domain),
                }
            )

    for contact in contact_records[:10]:
        if contact["relationship"] == "unrelated_third_party":
            add_feature(
                features,
                "contact.identity_domain_mismatch",
                "suspicious",
                "medium",
                {
                    "contact": contact["contact"],
                    "contact_domain": contact["contact_domain"],
                    "page_domain": domain,
                    "relationship": contact["relationship"],
                    "kind": contact["kind"],
                },
                f"Visible contact identity {contact['contact']} does not match page domain {domain}.",
            )
            break

    if contact_records and not any(feature["id"] == "contact.identity_domain_mismatch" for feature in features):
        matching = [
            contact
            for contact in contact_records
            if contact["relationship"] in {"same_registrable_domain", "same_organization_or_alias"}
        ]
        if matching:
            add_feature(
                features,
                "support.contact_domain_match",
                "benign",
                "medium",
                {"domain": domain, "count": len(matching), "kinds": sorted({item["kind"] for item in matching})},
                "Support or contact information matches the page domain.",
            )


def extract_link_features(page: dict[str, Any], args: argparse.Namespace, features: list[dict[str, Any]]) -> None:
    anchors = all_anchor_items(page)
    base_url = norm_text(page.get("final_url") or page.get("url"))
    base_domain = page_domain(page)
    null_hrefs = []
    external = 0
    internal_functional = 0
    brand_mismatches: list[dict[str, str]] = []
    idp_domains: set[str] = set()
    contact_matches = 0
    credential_context = has_credential_form(page)

    for anchor in anchors:
        href = norm_text(anchor.get("href"))
        text = anchor_text(anchor)
        href_l = href.lower()
        if not href or href_l == "#" or href_l.startswith("javascript:void"):
            null_hrefs.append(href or "<empty>")
            continue

        href_domain = absolute_domain(base_url, href)
        rel = relationship(base_domain, href_domain)
        if href_domain and rel not in {"same_registrable_domain", "same_organization_or_alias"}:
            external += 1
        if is_meaningful_nav_anchor(anchor, base_url, base_domain):
            internal_functional += 1

        if rel == "known_identity_provider" and credential_context:
            idp_domains.add(href_domain)

        if any(term in lower_text(text) for term in {"contact", "support", "help", "privacy", "legal"}):
            if rel in {"same_registrable_domain", "same_organization_or_alias"}:
                contact_matches += 1

        for claim in extract_brand_claims(text):
            if href_domain and not brand_matches_domain(claim["brand"], href_domain):
                if rel in {
                    "known_identity_provider",
                    "known_infrastructure_provider",
                    "known_payment_or_app_store",
                } or domain_in_set(href_domain, KNOWN_SOCIAL_OR_PROFILE_DOMAINS):
                    continue
                brand_mismatches.append({"brand": claim["brand"], "href_domain": href_domain, "relationship": rel})
                break

    total = len(anchors)
    if null_hrefs:
        add_feature(
            features,
            "link.null_or_void_anchors",
            "suspicious",
            "low",
            {"count": len(null_hrefs), "total": total, "examples": null_hrefs[:5]},
            f"Page includes {len(null_hrefs)} null or void link(s).",
        )

    for mismatch in brand_mismatches[:3]:
        add_feature(
            features,
            "link.brand_text_domain_mismatch",
            "suspicious",
            "low",
            mismatch,
            f"Link text suggests {mismatch['brand']}, but the link points to {mismatch['href_domain']}.",
        )

    if total and external / total >= args.high_external_anchor_ratio and has_credential_form(page):
        add_feature(
            features,
            "link.high_external_anchor_ratio",
            "suspicious",
            "low",
            {
                "external_count": external,
                "total_count": total,
                "ratio": round(external / total, 4),
                "threshold": args.high_external_anchor_ratio,
                "threshold_source": "cli_uncalibrated",
            },
            f"{external} of {total} links point to external domains.",
        )

    if internal_functional >= 3:
        add_feature(
            features,
            "navigation.functional_internal_links",
            "benign",
            "low",
            {"count": internal_functional},
            "Page has functional internal navigation links.",
        )

    if idp_domains:
        domain = sorted(idp_domains)[0]
        add_feature(
            features,
            "identity.provider_expected_domain",
            "benign",
            "medium",
            {"provider_domain": domain},
            f"Authentication uses an expected identity provider domain: {domain}.",
        )

    if contact_matches and not has_feature(features, "support.contact_domain_match"):
        add_feature(
            features,
            "support.contact_domain_match",
            "benign",
            "medium",
            {"domain": base_domain, "count": contact_matches},
            "Support or contact information matches the page domain.",
        )


def extract_iframe_features(page: dict[str, Any], features: list[dict[str, Any]]) -> None:
    for iframe in all_iframe_items(page):
        style = lower_text(iframe.get("style"))
        width = lower_text(iframe.get("width"))
        height = lower_text(iframe.get("height"))
        if "display:none" in style.replace(" ", "") or "visibility:hidden" in style.replace(" ", "") or width == "0" or height == "0":
            add_feature(
                features,
                "iframe.hidden_iframe",
                "suspicious",
                "low",
                {"src": norm_text(iframe.get("src"))[:300]},
                "Page contains a hidden iframe.",
            )
            return


def extract_uncertainty_features(page: dict[str, Any], features: list[dict[str, Any]]) -> None:
    text = lower_text(page.get("visible_text"))
    title = lower_text(page.get("title"))
    forms = all_form_items(page)
    anchors = all_anchor_items(page)
    has_credential = has_credential_form(page)
    claims = extract_brand_claims(page_identity_text(page))
    resources = page.get("resources") or {}
    script_count = len(resources.get("script_src_sample") or [])

    title_is_generic_or_empty = not title or title in GENERIC_TITLES
    if len(text) < 80 and not forms and len(anchors) < 2 and title_is_generic_or_empty:
        add_feature(
            features,
            "page.low_semantic_content",
            "uncertain",
            "medium",
            {
                "visible_text_length": len(text),
                "form_count": len(forms),
                "anchor_count": len(anchors),
                "title": title,
                "title_is_generic_or_empty": title_is_generic_or_empty,
                "visible_text_length_threshold": 80,
                "threshold_source": "hardcoded_crude_proxy",
            },
            "Page has too little semantic content to support a confident verdict.",
        )

    if has_credential and not claims:
        add_feature(
            features,
            "page.generic_login_without_brand_claim",
            "uncertain",
            "medium",
            {"form_count": len(forms)},
            "Page contains a generic login form without a clear brand claim.",
        )

    javascript_prompt = any(
        phrase in text
        for phrase in {
            "enable javascript",
            "requires javascript",
            "please enable javascript",
            "javascript is required",
        }
    )
    if (len(text) < 200 and script_count >= 5 and not forms and len(anchors) < 3) or javascript_prompt:
        add_feature(
            features,
            "page.rendering_incomplete_or_script_dependent",
            "uncertain",
            "medium",
            {
                "visible_text_length": len(text),
                "visible_text_length_threshold": 200,
                "script_sample_count": script_count,
                "script_sample_count_threshold": 5,
                "form_count": len(forms),
                "anchor_count": len(anchors),
                "javascript_prompt": javascript_prompt,
                "threshold_source": "hardcoded_crude_proxy",
            },
            "Page appears script-dependent or incompletely rendered in the capture.",
        )

    if title in {"403 forbidden", "404 not found", "access denied", "privacy error"}:
        add_feature(
            features,
            "capture.non_html_or_access_blocked",
            "uncertain",
            "high",
            {"title": title},
            "Capture is an access-blocked or non-HTML page.",
        )

    interstitial = matched_terms(text, INTERSTITIAL_TERMS)
    if interstitial and not has_credential:
        add_feature(
            features,
            "page.interstitial_without_credential_collection",
            "uncertain",
            "high",
            {"terms": interstitial},
            "Page is an interstitial and does not provide enough target-site context.",
        )


def extract_login_benign_and_missing_help(page: dict[str, Any], features: list[dict[str, Any]]) -> None:
    if not has_credential_form(page):
        return
    text = lower_text(page_identity_text(page))
    anchors_and_buttons = []
    for anchor in all_anchor_items(page):
        anchors_and_buttons.append(lower_text(anchor.get("text")))
        anchors_and_buttons.append(lower_text(anchor.get("title")))
        anchors_and_buttons.append(lower_text(anchor.get("aria_label")))
    for form in all_form_items(page):
        for button in form.get("buttons") or []:
            anchors_and_buttons.append(lower_text(button))
    nav_text = " ".join(part for part in anchors_and_buttons if part)
    terms = matched_terms(nav_text or text, HELP_TERMS)
    if not terms:
        add_feature(
            features,
            "login.missing_recovery_or_help_flow",
            "suspicious",
            "low",
            {"checked_terms": sorted(HELP_TERMS)},
            "Login form lacks normal recovery, help, or account-support links.",
        )


def dedupe_features(features: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str]] = set()
    deduped: list[dict[str, Any]] = []
    for feature in features:
        key = (feature["id"], json.dumps(feature.get("value", {}), sort_keys=True, ensure_ascii=False))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(feature)
    return deduped


def extract_features(record: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    page = record.get("input") or {}
    features: list[dict[str, Any]] = []
    extract_url_features(page, args, features)
    extract_redirect_features(page, args, features)
    extract_meta_features(page, features)
    extract_form_features(page, features)
    extract_brand_features(page, features)
    extract_content_features(page, features)
    extract_link_features(page, args, features)
    extract_iframe_features(page, features)
    extract_login_benign_and_missing_help(page, features)
    extract_uncertainty_features(page, features)
    features = dedupe_features(features)
    if args.max_features_per_record:
        return features[: args.max_features_per_record]
    return features


def input_total(path: Path) -> int | None:
    if str(path) == "-":
        return None
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    handle = sys.stdin if str(path) == "-" else path.open("r", encoding="utf-8")
    try:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
    finally:
        if handle is not sys.stdin:
            handle.close()


def open_output(path: Path | None):
    if path is None or str(path) == "-":
        return sys.stdout
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.open("w", encoding="utf-8")


def main() -> int:
    args = parse_args()
    output = open_output(args.output)
    close_output = output is not sys.stdout
    total = input_total(args.input)
    written = 0
    try:
        for record in tqdm(
            iter_jsonl(args.input),
            total=total,
            desc="extract_features",
            unit="record",
            disable=args.no_progress,
            file=sys.stderr,
        ):
            out_record = {
                "id": record.get("id"),
                "label": record.get("label"),
                "source": record.get("source"),
                "input": record.get("input") or {},
                "measurements": extract_measurements(record.get("input") or {}),
                "features": extract_features(record, args),
            }
            output.write(json.dumps(out_record, ensure_ascii=False, separators=(",", ":")))
            output.write("\n")
            written += 1
        if close_output:
            print(f"Wrote {written} JSONL record(s) to {args.output}", file=sys.stderr)
        return 0
    finally:
        if close_output:
            output.close()


if __name__ == "__main__":
    raise SystemExit(main())
