#!/usr/bin/env python3
"""Extract modular feature families from Stage A JSONL records.

This script follows the Stage B output shape used in extract_features_jsonl.py
but uses a registry of decoupled feature extractors so individual features can
be added, removed, or filtered without editing a monolithic extraction flow.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from difflib import SequenceMatcher
from functools import cached_property, lru_cache
import json
import re
import sys
from pathlib import Path
from typing import Any, Callable, Iterable
from urllib.parse import urljoin, urlparse

import tldextract

try:
    from tqdm import tqdm
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    def tqdm(iterable, **_: Any):
        return iterable


extract_tld = tldextract.TLDExtract(suffix_list_urls=())

EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)

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
    "account",
    "e-mail",
    "email",
    "login",
    "log in",
    "password",
    "security",
    "sign in",
    "signin",
    "user id",
    "username",
    "verify",
}

LOGIN_TERMS = {
    "login",
    "log in",
    "sign in",
    "signin",
    "authenticate",
    "webmail",
    "portal",
}

SIGNUP_TERMS = {
    "sign up",
    "signup",
    "register",
    "registration",
    "create account",
    "join now",
}

RECOVERY_TERMS = {
    "forgot password",
    "reset password",
    "recover account",
    "password recovery",
    "trouble signing in",
}

ACCOUNT_TERMS = {
    "account",
    "my account",
    "account settings",
    "manage account",
    "profile",
    "dashboard",
}

SUPPORT_TERMS = {
    "contact",
    "customer service",
    "faq",
    "help",
    "legal",
    "privacy",
    "support",
}

PRIVACY_TERMS = {
    "cookie policy",
    "cookie settings",
    "privacy",
    "privacy notice",
    "privacy policy",
    "terms",
    "terms of service",
    "terms of use",
}

COOKIE_TERMS = {
    "cookie",
    "cookie consent",
    "cookie policy",
    "cookie settings",
    "accept cookies",
}

DISCLOSURE_TERMS = {
    "affiliate disclosure",
    "disclaimer",
    "disclosure",
    "risk disclosure",
    "sponsored",
}

VERIFICATION_TERMS = {
    "account verification",
    "confirm your account",
    "human verification",
    "identity verification",
    "verification",
    "verify account",
    "verify identity",
}

CAPTCHA_TERMS = {
    "captcha",
    "i'm not a robot",
    "recaptcha",
    "turnstile",
    "verify you are human",
}

SECURITY_TERMS = {
    "protected by",
    "secure",
    "secure login",
    "security alert",
    "security challenge",
    "security notice",
    "ssl",
    "trusted",
}

URGENCY_TERMS = {
    "act now",
    "confirm within",
    "expire",
    "expired",
    "immediately",
    "limited time",
    "locked",
    "suspended",
    "unusual activity",
    "urgent",
    "verify immediately",
}

TRUST_TERMS = {
    "100% secure",
    "encrypted",
    "guaranteed",
    "official",
    "protected",
    "safe",
    "trusted",
    "verified",
}

DOWNLOAD_TERMS = {
    "app download",
    "download",
    "download file",
    "download now",
    "download the app",
    "get the app",
}

INSTALLER_TERMS = {
    "apk",
    "installer",
    "install now",
    "installation",
    "setup file",
    "update your app",
}

PAYMENT_TERMS = {
    "bill",
    "invoice",
    "payment",
    "pay now",
    "remittance",
    "settlement",
    "transfer",
    "wire",
}

CHECKOUT_TERMS = {
    "add to cart",
    "billing address",
    "buy now",
    "cart",
    "checkout",
    "order summary",
    "shipping",
}

FILE_LURE_TERMS = {
    "document review",
    "download file",
    "fax",
    "invoice",
    "payment",
    "remittance",
    "review document",
    "secure document",
    "shared document",
    "shared file",
    "voice mail",
    "voicemail",
}

ARTICLE_TERMS = {
    "article",
    "breaking news",
    "editorial",
    "news",
    "opinion",
    "published",
    "reporter",
}

BUSINESS_TERMS = {
    "business",
    "catalog",
    "company",
    "our services",
    "pricing",
    "solutions",
    "why choose us",
}

PROFILE_TERMS = {
    "about us",
    "company profile",
    "organization",
    "our profile",
    "profile",
    "team",
}

DIRECTORY_TERMS = {
    "categories",
    "directory",
    "listing",
    "locations",
    "services",
}

CHAT_TERMS = {
    "chat",
    "chat with us",
    "live chat",
    "message us",
    "support chat",
}

COMMENT_TERMS = {
    "comments",
    "followers",
    "ratings",
    "reviews",
    "subscribers",
    "testimonials",
}

ERROR_TITLES = {
    "403 forbidden",
    "404 not found",
    "access denied",
    "privacy error",
}

SCRIPT_DEPENDENCY_TERMS = {
    "enable javascript",
    "javascript is required",
    "please enable javascript",
    "requires javascript",
}

FREE_EMAIL_DOMAINS = {
    "aol.com",
    "gmail.com",
    "hotmail.com",
    "icloud.com",
    "mail.com",
    "outlook.com",
    "proton.me",
    "protonmail.com",
    "yahoo.com",
}

CHAT_PROVIDER_DOMAINS = {
    "crisp.chat",
    "drift.com",
    "freshchat.com",
    "intercom.com",
    "livechatinc.com",
    "tawk.to",
    "zendesk.com",
}

CAPTCHA_PROVIDER_DOMAINS = {
    "challenges.cloudflare.com",
    "cloudflare.com",
    "google.com",
    "gstatic.com",
    "hcaptcha.com",
}

ANALYTICS_PROVIDER_DOMAINS = {
    "doubleclick.net",
    "facebook.net",
    "google-analytics.com",
    "googletagmanager.com",
    "hotjar.com",
    "segment.com",
}

STORE_PROVIDER_DOMAINS = {
    "bigcommerce.com",
    "myshopify.com",
    "shopify.com",
    "woocommerce.com",
}

SOCIAL_DOMAINS = {
    "facebook.com",
    "instagram.com",
    "linkedin.com",
    "threads.net",
    "tiktok.com",
    "twitter.com",
    "x.com",
    "youtube.com",
}

KNOWN_IDP_DOMAINS = {
    "accounts.google.com",
    "auth0.com",
    "duosecurity.com",
    "login.microsoftonline.com",
    "login.salesforce.com",
    "microsoftonline.com",
    "myworkday.com",
    "okta.com",
    "onelogin.com",
    "pingidentity.com",
    "pingone.com",
    "salesforce.com",
    "workday.com",
}

KNOWN_INFRA_DOMAINS = {
    "akamaihd.net",
    "akamaiedge.net",
    "amazonaws.com",
    "appspot.com",
    "azureedge.net",
    "cloudflare.com",
    "cloudfront.net",
    "firebaseapp.com",
    "github.io",
    "herokuapp.com",
    "netlify.app",
    "pages.dev",
    "squarespace.com",
    "vercel.app",
    "web.app",
    "wixsite.com",
}

KNOWN_PAYMENT_OR_APP_DOMAINS = {
    "adyen.com",
    "apps.apple.com",
    "braintreepayments.com",
    "checkout.com",
    "itunes.apple.com",
    "paypal.com",
    "play.google.com",
    "squareup.com",
    "stripe.com",
}

BRAND_ALIASES: dict[str, set[str]] = {
    "adobe": {"adobe", "acrobat"},
    "amazon": {"amazon", "aws"},
    "apple": {"apple", "apple id", "app store", "icloud"},
    "dhl": {"dhl"},
    "docuSign": {"docusign"},
    "dropbox": {"dropbox"},
    "fedex": {"fedex"},
    "google": {"gmail", "google", "google drive", "workspace"},
    "linkedin": {"linkedin"},
    "meta": {"facebook", "instagram", "meta", "whatsapp"},
    "microsoft": {"microsoft", "office 365", "office365", "onedrive", "outlook", "sharepoint", "webmail exchange"},
    "naver": {"naver"},
    "netflix": {"netflix"},
    "paypal": {"paypal"},
    "ups": {"ups"},
    "yahoo": {"yahoo"},
}

BRAND_DOMAINS: dict[str, set[str]] = {
    "adobe": {"adobe.com", "adobe.io"},
    "amazon": {"amazon.com", "amazonaws.com", "aws.amazon.com"},
    "apple": {"apple.com", "icloud.com", "mzstatic.com"},
    "dhl": {"dhl.com"},
    "docuSign": {"docusign.com", "docusign.net"},
    "dropbox": {"dropbox.com", "dropboxusercontent.com"},
    "fedex": {"fedex.com"},
    "google": {"gmail.com", "google.com", "googleapis.com", "googleusercontent.com", "gstatic.com"},
    "linkedin": {"linkedin.com"},
    "meta": {"facebook.com", "fbcdn.net", "instagram.com", "meta.com", "whatsapp.com"},
    "microsoft": {"live.com", "microsoft.com", "microsoftonline.com", "office.com", "office365.com", "onedrive.com", "outlook.com", "sharepoint.com"},
    "naver": {"naver.com"},
    "netflix": {"netflix.com"},
    "paypal": {"paypal.com", "paypalobjects.com"},
    "ups": {"ups.com"},
    "yahoo": {"yahoo.com"},
}

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

FEATURE_SUPERVISION: dict[str, dict[str, Any]] = {}


def norm_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def lower_text(value: Any) -> str:
    return norm_text(value).lower()


def hostname(url: str) -> str:
    return (urlparse(norm_text(url)).hostname or "").lower()


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


def domain_in_set(domain: str, known_domains: set[str]) -> bool:
    if not domain:
        return False
    return any(domain == item or domain.endswith("." + item) for item in known_domains)


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


def brand_matches_domain(brand: str, domain: str) -> bool:
    return domain_in_set(domain, BRAND_DOMAINS.get(brand, set()))


def known_provider_relationship(domain: str) -> str | None:
    domain = (domain or "").lower()
    registrable = registrable_domain(domain)
    if domain_in_set(domain, KNOWN_PAYMENT_OR_APP_DOMAINS) or domain_in_set(registrable, KNOWN_PAYMENT_OR_APP_DOMAINS):
        return "known_payment_or_app_store"
    if domain_in_set(domain, KNOWN_IDP_DOMAINS) or domain_in_set(registrable, KNOWN_IDP_DOMAINS):
        return "known_identity_provider"
    if domain_in_set(domain, KNOWN_INFRA_DOMAINS) or domain_in_set(registrable, KNOWN_INFRA_DOMAINS):
        return "known_infrastructure_provider"
    if domain_in_set(domain, SOCIAL_DOMAINS) or domain_in_set(registrable, SOCIAL_DOMAINS):
        return "known_social_or_profile_provider"
    if domain_in_set(domain, CHAT_PROVIDER_DOMAINS) or domain_in_set(registrable, CHAT_PROVIDER_DOMAINS):
        return "known_chat_provider"
    if domain_in_set(domain, CAPTCHA_PROVIDER_DOMAINS) or domain_in_set(registrable, CAPTCHA_PROVIDER_DOMAINS):
        return "known_captcha_provider"
    return None


def relationship(source_domain: str, target_domain: str) -> str:
    source_raw = (source_domain or "").lower()
    target_raw = (target_domain or "").lower()
    if not source_raw or not target_raw:
        return "unknown"
    source_registrable = registrable_domain(source_raw)
    target_registrable = registrable_domain(target_raw)
    if source_registrable and target_registrable and source_registrable == target_registrable:
        return "same_registrable_domain"
    if domain_in_set(target_raw, KNOWN_PAYMENT_OR_APP_DOMAINS) or domain_in_set(target_registrable, KNOWN_PAYMENT_OR_APP_DOMAINS):
        return "known_payment_or_app_store"
    if domain_in_set(target_raw, KNOWN_IDP_DOMAINS) or domain_in_set(target_registrable, KNOWN_IDP_DOMAINS):
        return "known_identity_provider"
    if domain_in_set(target_raw, KNOWN_INFRA_DOMAINS) or domain_in_set(target_registrable, KNOWN_INFRA_DOMAINS):
        return "known_infrastructure_provider"
    if domain_in_set(target_raw, SOCIAL_DOMAINS) or domain_in_set(target_registrable, SOCIAL_DOMAINS):
        return "known_social_or_profile_provider"
    if domain_in_set(target_raw, CHAT_PROVIDER_DOMAINS) or domain_in_set(target_registrable, CHAT_PROVIDER_DOMAINS):
        return "known_chat_provider"
    if same_known_brand(source_registrable or source_raw, target_registrable or target_raw):
        return "same_organization_or_alias"
    return "unrelated_third_party"


def matched_terms(text: str, terms: set[str]) -> list[str]:
    text_l = lower_text(text)
    found: list[str] = []
    for term in sorted(terms):
        pattern = r"\b" + re.escape(term).replace(r"\ ", r"\s+") + r"\b"
        if re.search(pattern, text_l):
            found.append(term)
    return found


def redact_email(value: str) -> str:
    def repl(match: re.Match[str]) -> str:
        email = match.group(0)
        _, _, domain = email.partition("@")
        return f"<email>@{domain}" if domain else "<email>"

    return EMAIL_RE.sub(repl, value)


def decode_label(label: str) -> str:
    try:
        return label.encode("ascii").decode("idna")
    except UnicodeError:
        return label


def skeletonize_label(value: str) -> str:
    value = decode_label(lower_text(value))
    value = value.translate(CONFUSABLE_TRANSLATION)
    value = re.sub(r"[^a-z0-9]+", "", value)
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
    label_skeletons: list[str] = []
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
                    "matched_alias": alias,
                    "matched_label": label_skeleton,
                    "match_type": "confusable_exact",
                    "target_brand": brand,
                }
            if alias_skeleton in label_skeleton and len(alias_skeleton) >= 6:
                return {
                    "domain": domain,
                    "hostname": host or domain,
                    "matched_alias": alias,
                    "matched_label": label_skeleton,
                    "match_type": "brand_embedded_in_domain",
                    "target_brand": brand,
                }
            length_delta = abs(len(label_skeleton) - len(alias_skeleton))
            if len(alias_skeleton) >= 7 and length_delta <= 2:
                similarity = SequenceMatcher(None, label_skeleton, alias_skeleton).ratio()
                if similarity >= 0.88:
                    return {
                        "domain": domain,
                        "hostname": host or domain,
                        "matched_alias": alias,
                        "matched_label": label_skeleton,
                        "match_type": "near_edit_distance",
                        "similarity": round(similarity, 4),
                        "target_brand": brand,
                    }
    return None


def make_feature(
    feature_id: str,
    direction: str,
    severity: str,
    value: dict[str, Any],
) -> dict[str, Any]:
    feature = {
        "id": feature_id,
        "direction": direction,
        "severity": severity,
        "value": value,
    }
    return feature


@dataclass(frozen=True)
class FeatureGroupSpec:
    group_id: str
    extractor: Callable[["PageContext"], dict[str, Any] | None]


@dataclass(frozen=True)
class SelectedFeatureGroup:
    extractor: Callable[["PageContext"], dict[str, Any] | None]
    group_id: str
    output_ids: tuple[str, ...]


FEATURE_REGISTRY: list[FeatureGroupSpec] = []


def register(feature_id: str) -> Callable[[Callable[["PageContext"], dict[str, Any] | None]], Callable[["PageContext"], dict[str, Any] | None]]:
    def decorator(func: Callable[["PageContext"], dict[str, Any] | None]) -> Callable[["PageContext"], dict[str, Any] | None]:
        FEATURE_REGISTRY.append(FeatureGroupSpec(group_id=feature_id, extractor=func))
        return func

    return decorator


@lru_cache(maxsize=1)
def load_group_output_ids() -> dict[str, tuple[str, ...]]:
    markdown_path = Path(__file__).with_name("new_features.md")
    fallback = {spec.group_id: (spec.group_id,) for spec in FEATURE_REGISTRY}
    if not markdown_path.exists():
        return fallback

    text = markdown_path.read_text(encoding="utf-8")
    blocks = []
    pattern = re.compile(r"(?ms)^\d+\. \*\*(.+?)\*\*  \n   Covers: (.+?)\n   Method:")
    for match in pattern.finditer(text):
        covers = tuple(re.findall(r"`([^`]+)`", match.group(2)))
        blocks.append(covers)

    if len(blocks) != len(FEATURE_REGISTRY):
        return fallback

    output_map: dict[str, tuple[str, ...]] = {}
    for spec, covers in zip(FEATURE_REGISTRY, blocks):
        output_map[spec.group_id] = covers or (spec.group_id,)
    return output_map


@dataclass(frozen=True)
class LinkInfo:
    text: str
    href: str
    domain: str
    relationship: str
    is_external: bool
    is_hash: bool
    is_javascript: bool
    is_mailto: bool
    is_tel: bool
    is_internal: bool


@dataclass(frozen=True)
class ResourceInfo:
    kind: str
    raw: str
    domain: str
    relationship: str


class PageContext:
    def __init__(self, page: dict[str, Any]):
        self.page = page

    @cached_property
    def url(self) -> str:
        return norm_text(self.page.get("url"))

    @cached_property
    def final_url(self) -> str:
        return norm_text(self.page.get("final_url") or self.url)

    @cached_property
    def title(self) -> str:
        return norm_text(self.page.get("title"))

    @cached_property
    def visible_text(self) -> str:
        return norm_text(self.page.get("visible_text"))

    @cached_property
    def page_text(self) -> str:
        return norm_text(f"{self.title} {self.visible_text}")

    @cached_property
    def title_l(self) -> str:
        return lower_text(self.title)

    @cached_property
    def visible_text_l(self) -> str:
        return lower_text(self.visible_text)

    @cached_property
    def page_text_l(self) -> str:
        return lower_text(self.page_text)

    @cached_property
    def domain(self) -> str:
        return registrable_domain(self.final_url or self.url)

    @cached_property
    def host(self) -> str:
        return hostname(self.final_url or self.url)

    @cached_property
    def hostname_label_count(self) -> int:
        return len([part for part in self.host.split(".") if part]) if self.host else 0

    @cached_property
    def forms(self) -> list[dict[str, Any]]:
        forms = self.page.get("forms") or {}
        if isinstance(forms, dict):
            return [item for item in (forms.get("items") or []) if isinstance(item, dict)]
        if isinstance(forms, list):
            return [item for item in forms if isinstance(item, dict)]
        return []

    @cached_property
    def anchors(self) -> list[dict[str, Any]]:
        anchors = self.page.get("anchors") or {}
        if isinstance(anchors, dict):
            return [item for item in (anchors.get("items") or []) if isinstance(item, dict)]
        if isinstance(anchors, list):
            return [item for item in anchors if isinstance(item, dict)]
        return []

    @cached_property
    def iframes(self) -> list[dict[str, Any]]:
        iframes = self.page.get("iframes") or {}
        if isinstance(iframes, dict):
            return [item for item in (iframes.get("items") or []) if isinstance(item, dict)]
        if isinstance(iframes, list):
            return [item for item in iframes if isinstance(item, dict)]
        return []

    @cached_property
    def redirects(self) -> list[dict[str, Any]]:
        redirects = self.page.get("redirects") or []
        return [item for item in redirects if isinstance(item, dict)]

    @cached_property
    def meta_items(self) -> list[dict[str, Any]]:
        meta = self.page.get("meta") or []
        return [item for item in meta if isinstance(item, dict)]

    @cached_property
    def resources(self) -> dict[str, list[str]]:
        raw = self.page.get("resources") or {}
        if not isinstance(raw, dict):
            return {}
        out: dict[str, list[str]] = {}
        for key in ("favicon_hrefs", "image_src_sample", "script_src_sample", "stylesheet_href_sample"):
            values = raw.get(key) or []
            out[key] = [norm_text(item) for item in values if norm_text(item)]
        return out

    @cached_property
    def meta_by_name(self) -> dict[str, list[str]]:
        data: dict[str, list[str]] = {}
        for item in self.meta_items:
            name = lower_text(item.get("name"))
            if not name:
                continue
            data.setdefault(name, []).append(norm_text(item.get("content")))
        return data

    @cached_property
    def meta_by_property(self) -> dict[str, list[str]]:
        data: dict[str, list[str]] = {}
        for item in self.meta_items:
            name = lower_text(item.get("property"))
            if not name:
                continue
            data.setdefault(name, []).append(norm_text(item.get("content")))
        return data

    @cached_property
    def meta_text(self) -> str:
        parts: list[str] = []
        for item in self.meta_items:
            parts.extend(
                [
                    norm_text(item.get("name")),
                    norm_text(item.get("property")),
                    norm_text(item.get("http_equiv")),
                    norm_text(item.get("content")),
                ]
            )
        return norm_text(" ".join(part for part in parts if part))

    @cached_property
    def page_text_with_meta(self) -> str:
        return norm_text(f"{self.page_text} {self.meta_text}")

    @cached_property
    def form_infos(self) -> list[dict[str, Any]]:
        infos: list[dict[str, Any]] = []
        for form in self.forms:
            inputs = [item for item in (form.get("inputs") or []) if isinstance(item, dict)]
            button_texts = [norm_text(button) for button in (form.get("buttons") or []) if norm_text(button)]
            text_parts: list[str] = [norm_text(form.get("text"))]
            for input_item in inputs:
                text_parts.extend(
                    [
                        norm_text(input_item.get("label")),
                        norm_text(input_item.get("placeholder")),
                        norm_text(input_item.get("name")),
                        norm_text(input_item.get("id")),
                        norm_text(input_item.get("aria_label")),
                    ]
                )
            text_parts.extend(button_texts)
            text = norm_text(" ".join(part for part in text_parts if part))
            action = norm_text(form.get("action"))
            action_domain = absolute_domain(self.final_url or self.url, action) if action else ""
            rel = relationship(self.domain, action_domain) if action else "unknown"
            infos.append(
                {
                    "action": action,
                    "action_domain": action_domain,
                    "buttons": button_texts,
                    "inputs": inputs,
                    "relationship": rel,
                    "text": text,
                }
            )
        return infos

    @cached_property
    def form_text(self) -> str:
        return norm_text(" ".join(info["text"] for info in self.form_infos if info["text"]))

    @cached_property
    def form_button_text(self) -> str:
        return norm_text(" ".join(button for info in self.form_infos for button in info["buttons"]))

    @cached_property
    def anchor_infos(self) -> list[LinkInfo]:
        infos: list[LinkInfo] = []
        for anchor in self.anchors:
            href = norm_text(anchor.get("href"))
            href_l = href.lower()
            text = norm_text(
                " ".join(
                    [
                        norm_text(anchor.get("text")),
                        norm_text(anchor.get("title")),
                        norm_text(anchor.get("aria_label")),
                    ]
                )
            )
            domain = absolute_domain(self.final_url or self.url, href) if href else ""
            rel = relationship(self.domain, domain) if domain else "unknown"
            is_hash = href_l == "#" or href_l.startswith("#")
            is_js = href_l.startswith("javascript:")
            is_mailto = href_l.startswith("mailto:")
            is_tel = href_l.startswith("tel:")
            is_internal = bool(domain and rel in {"same_registrable_domain", "same_organization_or_alias"}) or (
                href and not urlparse(href).scheme and not href.startswith("#")
            )
            is_external = bool(domain) and rel not in {
                "same_organization_or_alias",
                "same_registrable_domain",
            }
            infos.append(
                LinkInfo(
                    text=text,
                    href=href,
                    domain=domain,
                    relationship=rel,
                    is_external=is_external,
                    is_hash=is_hash,
                    is_javascript=is_js,
                    is_mailto=is_mailto,
                    is_tel=is_tel,
                    is_internal=is_internal,
                )
            )
        return infos

    @cached_property
    def anchor_text(self) -> str:
        return norm_text(" ".join(item.text for item in self.anchor_infos if item.text))

    @cached_property
    def anchor_domains(self) -> list[str]:
        return [item.domain for item in self.anchor_infos if item.domain]

    @cached_property
    def null_anchor_count(self) -> int:
        return sum(1 for item in self.anchor_infos if not item.href or item.is_hash or item.is_javascript)

    @cached_property
    def external_anchor_count(self) -> int:
        return sum(1 for item in self.anchor_infos if item.is_external)

    @cached_property
    def internal_anchor_count(self) -> int:
        return sum(1 for item in self.anchor_infos if item.is_internal)

    @cached_property
    def redirect_domains(self) -> list[str]:
        return [registrable_domain(item.get("url") or "") for item in self.redirects if item.get("url")]

    @cached_property
    def input_types(self) -> list[str]:
        types: list[str] = []
        for info in self.form_infos:
            for input_item in info["inputs"]:
                input_type = lower_text(input_item.get("type")) or "text"
                types.append(input_type)
        return types

    @cached_property
    def password_input_count(self) -> int:
        return sum(1 for item in self.input_types if item == "password")

    @cached_property
    def email_input_count(self) -> int:
        return sum(1 for item in self.input_types if item == "email")

    @cached_property
    def number_input_count(self) -> int:
        return sum(1 for item in self.input_types if item in {"number", "tel"})

    @cached_property
    def hidden_input_count(self) -> int:
        return sum(1 for item in self.input_types if item == "hidden")

    @cached_property
    def login_form_count(self) -> int:
        total = 0
        for info in self.form_infos:
            text = info["text"]
            if any(t == "password" for t in [lower_text(inp.get("type")) or "text" for inp in info["inputs"]]) and matched_terms(
                text, LOGIN_TERMS | CREDENTIAL_TERMS
            ):
                total += 1
        return total

    @cached_property
    def credential_form_count(self) -> int:
        total = 0
        for info in self.form_infos:
            text = info["text"]
            input_types = [lower_text(inp.get("type")) or "text" for inp in info["inputs"]]
            if "password" in input_types or matched_terms(text, CREDENTIAL_TERMS):
                total += 1
        return total

    @cached_property
    def page_brand_claims(self) -> list[dict[str, str]]:
        return extract_brand_claims(self.page_text_with_meta)

    @cached_property
    def title_brand_claims(self) -> list[dict[str, str]]:
        return extract_brand_claims(self.title)

    @cached_property
    def visible_brand_claims(self) -> list[dict[str, str]]:
        return extract_brand_claims(self.visible_text)

    @cached_property
    def meta_brand_claims(self) -> list[dict[str, str]]:
        return extract_brand_claims(self.meta_text)

    @cached_property
    def brand_names_observed(self) -> set[str]:
        return {claim["brand"] for claim in self.page_brand_claims}

    @cached_property
    def resource_infos(self) -> list[ResourceInfo]:
        infos: list[ResourceInfo] = []
        for kind, values in self.resources.items():
            for raw in values:
                domain = absolute_domain(self.final_url or self.url, raw)
                infos.append(
                    ResourceInfo(
                        kind=kind,
                        raw=raw,
                        domain=domain,
                        relationship=relationship(self.domain, domain) if domain else "unknown",
                    )
                )
        return infos

    @cached_property
    def resource_domains(self) -> list[str]:
        return [item.domain for item in self.resource_infos if item.domain]

    @cached_property
    def visible_emails(self) -> list[str]:
        return EMAIL_RE.findall(self.page_text)

    @cached_property
    def visible_email_domains(self) -> list[str]:
        domains: list[str] = []
        for email in self.visible_emails:
            _, _, domain = email.partition("@")
            reg = registrable_domain(domain)
            if reg:
                domains.append(reg)
        return domains

    @cached_property
    def measurements(self) -> dict[str, Any]:
        return {
            "anchor_count": len(self.anchor_infos),
            "credential_form_count": self.credential_form_count,
            "domain": self.domain,
            "external_anchor_count": self.external_anchor_count,
            "final_hostname": self.host,
            "form_count": len(self.form_infos),
            "hidden_input_count": self.hidden_input_count,
            "hostname_label_count": self.hostname_label_count,
            "iframe_count": len(self.iframes),
            "internal_anchor_count": self.internal_anchor_count,
            "login_form_count": self.login_form_count,
            "meta_count": len(self.meta_items),
            "null_anchor_count": self.null_anchor_count,
            "password_input_count": self.password_input_count,
            "redirect_count": len(self.redirects),
            "resource_count": len(self.resource_infos),
            "title_is_generic": self.title_l in GENERIC_TITLES,
            "url_length": len(self.url),
            "visible_email_count": len(self.visible_emails),
            "visible_text_length": len(self.visible_text_l),
        }


def absolute_domain(base_url: str, maybe_url: str) -> str:
    if not maybe_url:
        return ""
    return hostname(urljoin(base_url, maybe_url))


def count_brand_domains(domains: Iterable[str]) -> set[str]:
    brands: set[str] = set()
    for domain in domains:
        brand = brand_for_domain(domain)
        if brand:
            brands.add(brand)
    return brands


def text_has_any(text: str, terms: set[str]) -> list[str]:
    return matched_terms(text, terms)


def captcha_like_url(value: str) -> bool:
    value_l = lower_text(value)
    return any(token in value_l for token in {"captcha", "recaptcha", "turnstile", "challenge"})


@register("url.domain_structure")
def feature_url_domain_structure(ctx: PageContext) -> dict[str, Any] | None:
    if not ctx.domain:
        return None
    self_refs = sum(1 for domain in ctx.anchor_domains + ctx.resource_domains if registrable_domain(domain) == ctx.domain)
    suspicious = ctx.hostname_label_count >= 4
    return make_feature(
        "url.domain_structure",
        "suspicious" if suspicious else "neutral",
        "low",
        {
            "domain": ctx.domain,
            "hostname": ctx.host,
            "hostname_label_count": ctx.hostname_label_count,
            "resource_self_reference_count": self_refs,
        })


@register("page.http_login")
def feature_page_http_login(ctx: PageContext) -> dict[str, Any] | None:
    if urlparse(ctx.final_url).scheme != "http" or ctx.login_form_count == 0:
        return None
    return make_feature(
        "page.http_login",
        "suspicious",
        "high",
        {"final_url": ctx.final_url, "login_form_count": ctx.login_form_count})


@register("redirect.https_upgrade")
def feature_redirect_https_upgrade(ctx: PageContext) -> dict[str, Any] | None:
    if not ctx.url or not ctx.final_url:
        return None
    if urlparse(ctx.url).scheme == "http" and urlparse(ctx.final_url).scheme == "https":
        return make_feature(
            "redirect.https_upgrade",
            "benign",
            "low",
            {"start_url": ctx.url, "final_url": ctx.final_url, "redirect_count": len(ctx.redirects)})
    return None


@register("redirect.chain_length")
def feature_redirect_chain_length(ctx: PageContext) -> dict[str, Any] | None:
    if not ctx.redirects:
        return None
    severity = "medium" if len(ctx.redirects) >= 2 else "low"
    direction = "suspicious" if len(ctx.redirects) >= 2 else "neutral"
    return make_feature(
        "redirect.chain_length",
        direction,
        severity,
        {"redirect_count": len(ctx.redirects), "redirect_domains": ctx.redirect_domains})


@register("redirect.destination_change")
def feature_redirect_destination_change(ctx: PageContext) -> dict[str, Any] | None:
    start_domain = registrable_domain(ctx.url)
    if not start_domain or not ctx.domain or start_domain == ctx.domain:
        return None
    return make_feature(
        "redirect.destination_change",
        "suspicious",
        "medium",
        {"final_domain": ctx.domain, "redirect_domains": ctx.redirect_domains, "start_domain": start_domain})


@register("brand.explicit_claim")
def feature_brand_explicit_claim(ctx: PageContext) -> dict[str, Any] | None:
    if not ctx.page_brand_claims:
        return None
    brands = sorted({claim["brand"] for claim in ctx.page_brand_claims})
    matched = sorted({claim["matched"] for claim in ctx.page_brand_claims})
    return make_feature(
        "brand.explicit_claim",
        "neutral",
        "low",
        {"brands": brands, "matched_terms": matched})


@register("brand.domain_mismatch")
def feature_brand_domain_mismatch(ctx: PageContext) -> dict[str, Any] | None:
    mismatches = []
    for claim in ctx.page_brand_claims:
        if not brand_matches_domain(claim["brand"], ctx.domain):
            mismatches.append(claim)
    if not mismatches:
        return None
    claim = mismatches[0]
    return make_feature(
        "brand.domain_mismatch",
        "suspicious",
        "medium",
        {"claimed_brand": claim["brand"], "domain": ctx.domain, "matched": claim["matched"]})


@register("brand.lookalike_domain")
def feature_brand_lookalike_domain(ctx: PageContext) -> dict[str, Any] | None:
    lookalike = domain_lookalike(ctx.final_url or ctx.url or ctx.domain)
    if not lookalike:
        return None
    return make_feature(
        "brand.lookalike_domain",
        "suspicious",
        "medium",
        lookalike)


@register("branding.consistency")
def feature_branding_consistency(ctx: PageContext) -> dict[str, Any] | None:
    page_brands = {claim["brand"] for claim in ctx.page_brand_claims}
    visual_brand_domains = [
        item.domain
        for item in ctx.resource_infos
        if item.domain and item.kind in {"favicon_hrefs", "image_src_sample"}
    ]
    resource_brands = count_brand_domains(visual_brand_domains)
    if page_brands:
        resource_brands &= page_brands
    sources = {
        "page": page_brands,
        "meta": {claim["brand"] for claim in ctx.meta_brand_claims},
        "resources": resource_brands,
    }
    observed = {brand for brands in sources.values() for brand in brands}
    if len(observed) < 2:
        return None
    consistent = len(observed) == 1
    return make_feature(
        "branding.consistency",
        "benign" if consistent else "suspicious",
        "low" if consistent else "medium",
        {
            "meta_brands": sorted(sources["meta"]),
            "page_brands": sorted(sources["page"]),
            "resource_brands": sorted(sources["resources"]),
        })


@register("footer.ownership_claim")
def feature_footer_ownership_claim(ctx: PageContext) -> dict[str, Any] | None:
    match = re.search(r"(copyright|\(c\)|©|all rights reserved|trademark).{0,120}", ctx.page_text, re.I)
    if not match:
        return None
    snippet = norm_text(match.group(0))
    claims = extract_brand_claims(snippet)
    direction = "neutral"
    if claims and not brand_matches_domain(claims[0]["brand"], ctx.domain):
        direction = "suspicious"
    return make_feature(
        "footer.ownership_claim",
        direction,
        "low",
        {"domain": ctx.domain, "snippet": snippet})


@register("form.presence")
def feature_form_presence(ctx: PageContext) -> dict[str, Any] | None:
    if not ctx.forms:
        return make_feature(
            "form.presence",
            "neutral",
            "low",
            {"form_count": 0})
    return make_feature(
        "form.presence",
        "neutral",
        "low",
        {"form_count": len(ctx.forms)})


@register("form.login_presence")
def feature_form_login_presence(ctx: PageContext) -> dict[str, Any] | None:
    if ctx.login_form_count == 0:
        return None
    return make_feature(
        "form.login_presence",
        "suspicious",
        "medium",
        {"login_form_count": ctx.login_form_count})


@register("form.credential_collection")
def feature_form_credential_collection(ctx: PageContext) -> dict[str, Any] | None:
    terms = matched_terms(f"{ctx.form_text} {ctx.visible_text}", CREDENTIAL_TERMS)
    if ctx.credential_form_count == 0 and not terms:
        return None
    return make_feature(
        "form.credential_collection",
        "suspicious",
        "high",
        {
            "credential_form_count": ctx.credential_form_count,
            "email_input_count": ctx.email_input_count,
            "matched_terms": terms,
            "password_input_count": ctx.password_input_count,
        })


@register("form.password_cues")
def feature_form_password_cues(ctx: PageContext) -> dict[str, Any] | None:
    if ctx.password_input_count == 0:
        return None
    return make_feature(
        "form.password_cues",
        "suspicious",
        "high",
        {"password_input_count": ctx.password_input_count})


@register("form.action_external_mismatch")
def feature_form_action_external_mismatch(ctx: PageContext) -> dict[str, Any] | None:
    hits = [
        info
        for info in ctx.form_infos
        if info["action_domain"]
        and info["relationship"] not in {"same_organization_or_alias", "same_registrable_domain", "known_identity_provider"}
    ]
    if not hits:
        return None
    item = hits[0]
    return make_feature(
        "form.action_external_mismatch",
        "suspicious",
        "medium",
        {"action": item["action"], "action_domain": item["action_domain"], "relationship": item["relationship"]})


@register("form.action_same_site")
def feature_form_action_same_site(ctx: PageContext) -> dict[str, Any] | None:
    count = sum(1 for info in ctx.form_infos if info["action_domain"] and info["relationship"] in {"same_organization_or_alias", "same_registrable_domain"})
    if count == 0:
        return None
    return make_feature(
        "form.action_same_site",
        "benign",
        "low",
        {"count": count})


@register("form.action_placeholder_or_self_submit")
def feature_form_action_placeholder_or_self_submit(ctx: PageContext) -> dict[str, Any] | None:
    findings: list[str] = []
    for info in ctx.form_infos:
        action_l = lower_text(info["action"])
        if not action_l or action_l == "about:blank":
            findings.append("blank")
        elif info["action"] == ctx.final_url or info["action"] == ctx.url:
            findings.append("self_submit")
        elif action_l in {"#", "javascript:void(0)", "javascript:void(0);"}:
            findings.append("placeholder")
    if not findings:
        return None
    return make_feature(
        "form.action_placeholder_or_self_submit",
        "uncertain",
        "low",
        {"count": len(findings), "findings": findings[:10]})


@register("link.recovery")
def feature_link_recovery(ctx: PageContext) -> dict[str, Any] | None:
    terms = matched_terms(f"{ctx.anchor_text} {ctx.form_button_text}", RECOVERY_TERMS)
    if not terms:
        return None
    return make_feature(
        "link.recovery",
        "benign",
        "low",
        {"matched_terms": terms})


@register("link.signup")
def feature_link_signup(ctx: PageContext) -> dict[str, Any] | None:
    terms = matched_terms(f"{ctx.anchor_text} {ctx.form_button_text}", SIGNUP_TERMS)
    if not terms:
        return None
    offsite_count = sum(1 for item in ctx.anchor_infos if item.is_external and matched_terms(item.text, SIGNUP_TERMS))
    return make_feature(
        "link.signup",
        "neutral",
        "low",
        {"matched_terms": terms, "offsite_target_count": offsite_count})


@register("navigation.account_actions")
def feature_navigation_account_actions(ctx: PageContext) -> dict[str, Any] | None:
    terms = matched_terms(f"{ctx.anchor_text} {ctx.visible_text}", ACCOUNT_TERMS)
    if not terms:
        return None
    return make_feature(
        "navigation.account_actions",
        "neutral",
        "low",
        {"matched_terms": terms})


@register("contact.presence")
def feature_contact_presence(ctx: PageContext) -> dict[str, Any] | None:
    mailto_count = sum(1 for item in ctx.anchor_infos if item.is_mailto)
    tel_count = sum(1 for item in ctx.anchor_infos if item.is_tel)
    contact_forms = 0
    for info in ctx.form_infos:
        types = {lower_text(inp.get("type")) or "text" for inp in info["inputs"]}
        if "password" not in types and (
            "email" in types
            or "tel" in types
            or matched_terms(info["text"], {"message", "contact", "quote", "phone"})
        ):
            contact_forms += 1
    methods = int(bool(ctx.visible_emails)) + int(mailto_count > 0) + int(tel_count > 0) + int(contact_forms > 0)
    if methods == 0:
        return None
    return make_feature(
        "contact.presence",
        "benign",
        "low",
        {
            "contact_form_count": contact_forms,
            "mailto_count": mailto_count,
            "method_count": methods,
            "tel_count": tel_count,
            "visible_email_count": len(ctx.visible_emails),
        })


@register("link.support_contact_legal")
def feature_link_support_contact_legal(ctx: PageContext) -> dict[str, Any] | None:
    matched = []
    internal = 0
    external = 0
    for item in ctx.anchor_infos:
        terms = matched_terms(item.text, SUPPORT_TERMS)
        if not terms:
            continue
        matched.extend(terms)
        if item.is_internal:
            internal += 1
        elif item.is_external:
            external += 1
    if not matched:
        return None
    direction = "benign" if internal >= external else "neutral"
    return make_feature(
        "link.support_contact_legal",
        direction,
        "low",
        {"external_count": external, "internal_count": internal, "matched_terms": sorted(set(matched))})


@register("contact.identity_mismatch")
def feature_contact_identity_mismatch(ctx: PageContext) -> dict[str, Any] | None:
    mismatched = [domain for domain in ctx.visible_email_domains if domain and relationship(ctx.domain, domain) == "unrelated_third_party"]
    if not mismatched:
        return None
    return make_feature(
        "contact.identity_mismatch",
        "suspicious",
        "medium",
        {"email_domains": sorted(set(mismatched)), "page_domain": ctx.domain})


@register("contact.free_provider_email")
def feature_contact_free_provider_email(ctx: PageContext) -> dict[str, Any] | None:
    free_domains = [domain for domain in ctx.visible_email_domains if domain in FREE_EMAIL_DOMAINS]
    if not free_domains:
        return None
    return make_feature(
        "contact.free_provider_email",
        "neutral",
        "low",
        {"email_domains": sorted(set(free_domains))})


@register("anchors.inventory")
def feature_anchors_inventory(ctx: PageContext) -> dict[str, Any] | None:
    return make_feature(
        "anchors.inventory",
        "neutral",
        "low",
        {"anchor_count": len(ctx.anchor_infos)})


@register("anchors.nonfunctional")
def feature_anchors_nonfunctional(ctx: PageContext) -> dict[str, Any] | None:
    if ctx.null_anchor_count == 0:
        return None
    return make_feature(
        "anchors.nonfunctional",
        "neutral",
        "low",
        {"null_anchor_count": ctx.null_anchor_count})


@register("navigation.internal")
def feature_navigation_internal(ctx: PageContext) -> dict[str, Any] | None:
    if ctx.internal_anchor_count < 3:
        return None
    return make_feature(
        "navigation.internal",
        "benign",
        "low",
        {"internal_anchor_count": ctx.internal_anchor_count})


@register("links.external_offsite")
def feature_links_external_offsite(ctx: PageContext) -> dict[str, Any] | None:
    if ctx.external_anchor_count == 0:
        return None
    return make_feature(
        "links.external_offsite",
        "neutral",
        "low",
        {"external_anchor_count": ctx.external_anchor_count, "total_anchor_count": len(ctx.anchor_infos)})


@register("links.external_account_profile_social")
def feature_links_external_account_profile_social(ctx: PageContext) -> dict[str, Any] | None:
    hits = [
        item.domain
        for item in ctx.anchor_infos
        if item.domain and (domain_in_set(item.domain, SOCIAL_DOMAINS) or item.relationship == "known_social_or_profile_provider")
    ]
    if not hits:
        return None
    return make_feature(
        "links.external_account_profile_social",
        "neutral",
        "low",
        {"domains": sorted(set(hits))})


@register("link.text_destination_mismatch")
def feature_link_text_destination_mismatch(ctx: PageContext) -> dict[str, Any] | None:
    mismatches = []
    for item in ctx.anchor_infos:
        claims = extract_brand_claims(item.text)
        if not claims or not item.domain:
            continue
        claim = claims[0]
        if not brand_matches_domain(claim["brand"], item.domain):
            mismatches.append({"brand": claim["brand"], "domain": item.domain, "text": item.text})
    if not mismatches:
        return None
    mismatch = mismatches[0]
    return make_feature(
        "link.text_destination_mismatch",
        "suspicious",
        "medium",
        mismatch)


@register("navigation.directory_store")
def feature_navigation_directory_store(ctx: PageContext) -> dict[str, Any] | None:
    terms = matched_terms(ctx.anchor_text, DIRECTORY_TERMS | {"store", "shop", "products"})
    if not terms:
        return None
    return make_feature(
        "navigation.directory_store",
        "neutral",
        "low",
        {"matched_terms": terms})


@register("policy.privacy_terms")
def feature_policy_privacy_terms(ctx: PageContext) -> dict[str, Any] | None:
    matched = matched_terms(f"{ctx.page_text} {ctx.anchor_text}", PRIVACY_TERMS)
    if not matched:
        return None
    placeholders = sum(
        1
        for item in ctx.anchor_infos
        if matched_terms(item.text, PRIVACY_TERMS) and (not item.href or item.is_hash or item.is_javascript)
    )
    return make_feature(
        "policy.privacy_terms",
        "neutral",
        "low",
        {"matched_terms": matched, "placeholder_link_count": placeholders})


@register("cookie.notice")
def feature_cookie_notice(ctx: PageContext) -> dict[str, Any] | None:
    matched = matched_terms(ctx.page_text, COOKIE_TERMS)
    if not matched:
        return None
    return make_feature(
        "cookie.notice",
        "benign",
        "low",
        {"matched_terms": matched})


@register("disclosure.present")
def feature_disclosure_present(ctx: PageContext) -> dict[str, Any] | None:
    matched = matched_terms(ctx.page_text, DISCLOSURE_TERMS)
    if not matched:
        return None
    return make_feature(
        "disclosure.present",
        "neutral",
        "low",
        {"matched_terms": matched})


@register("verification.challenge_language")
def feature_verification_challenge_language(ctx: PageContext) -> dict[str, Any] | None:
    matched = matched_terms(f"{ctx.page_text} {ctx.form_text} {ctx.form_button_text}", VERIFICATION_TERMS)
    if not matched:
        return None
    return make_feature(
        "verification.challenge_language",
        "suspicious",
        "medium",
        {"matched_terms": matched})


@register("challenge.captcha_or_antibot")
def feature_challenge_captcha_or_antibot(ctx: PageContext) -> dict[str, Any] | None:
    matched = matched_terms(f"{ctx.page_text} {ctx.anchor_text}", CAPTCHA_TERMS)
    provider_domains: set[str] = set()
    for item in ctx.resource_infos:
        if item.domain and captcha_like_url(item.raw) and (
            domain_in_set(item.domain, CAPTCHA_PROVIDER_DOMAINS) or known_provider_relationship(item.domain) == "known_captcha_provider"
        ):
            provider_domains.add(item.domain)
    for iframe in ctx.iframes:
        src = norm_text(iframe.get("src"))
        domain = absolute_domain(ctx.final_url or ctx.url, src)
        if domain and captcha_like_url(src) and (
            domain_in_set(domain, CAPTCHA_PROVIDER_DOMAINS) or known_provider_relationship(domain) == "known_captcha_provider"
        ):
            provider_domains.add(domain)
    if not matched and not provider_domains:
        return None
    return make_feature(
        "challenge.captcha_or_antibot",
        "neutral",
        "low",
        {"matched_terms": matched, "provider_domains": sorted(provider_domains)})


@register("security.claim_language")
def feature_security_claim_language(ctx: PageContext) -> dict[str, Any] | None:
    matched = matched_terms(f"{ctx.page_text} {ctx.form_text} {ctx.form_button_text}", SECURITY_TERMS)
    if not matched:
        return None
    return make_feature(
        "security.claim_language",
        "neutral",
        "low",
        {"matched_terms": matched})


@register("urgency.language")
def feature_urgency_language(ctx: PageContext) -> dict[str, Any] | None:
    matched = matched_terms(f"{ctx.page_text} {ctx.form_text} {ctx.form_button_text}", URGENCY_TERMS)
    if not matched:
        return None
    return make_feature(
        "urgency.language",
        "suspicious",
        "medium",
        {"matched_terms": matched})


@register("trust.reassurance_language")
def feature_trust_reassurance_language(ctx: PageContext) -> dict[str, Any] | None:
    matched = matched_terms(ctx.page_text, TRUST_TERMS)
    if not matched:
        return None
    return make_feature(
        "trust.reassurance_language",
        "neutral",
        "low",
        {"matched_terms": matched})


@register("language.mix_or_localization_mismatch")
def feature_language_mix_or_localization_mismatch(ctx: PageContext) -> dict[str, Any] | None:
    scripts = {
        "latin": bool(re.search(r"[A-Za-z]", ctx.page_text)),
        "cyrillic": bool(re.search(r"[\u0400-\u04FF]", ctx.page_text)),
        "arabic": bool(re.search(r"[\u0600-\u06FF]", ctx.page_text)),
        "cjk": bool(re.search(r"[\u3040-\u30ff\u3400-\u9fff]", ctx.page_text)),
    }
    active = [name for name, present in scripts.items() if present]
    if len(active) < 2:
        return None
    return make_feature(
        "language.mix_or_localization_mismatch",
        "uncertain",
        "low",
        {"active_scripts": active})


@register("content.download_prompt")
def feature_content_download_prompt(ctx: PageContext) -> dict[str, Any] | None:
    matched = matched_terms(f"{ctx.page_text} {ctx.anchor_text} {ctx.form_button_text}", DOWNLOAD_TERMS)
    if not matched:
        return None
    return make_feature(
        "content.download_prompt",
        "neutral",
        "low",
        {"matched_terms": matched})


@register("content.installer_prompt")
def feature_content_installer_prompt(ctx: PageContext) -> dict[str, Any] | None:
    matched = matched_terms(f"{ctx.page_text} {ctx.anchor_text}", INSTALLER_TERMS)
    install_domains = [
        item.domain
        for item in ctx.anchor_infos
        if item.domain and relationship(ctx.domain, item.domain) == "known_payment_or_app_store"
    ]
    if not matched and not install_domains:
        return None
    return make_feature(
        "content.installer_prompt",
        "neutral",
        "low",
        {"install_domains": sorted(set(install_domains)), "matched_terms": matched})


@register("content.payment_request_language")
def feature_content_payment_request_language(ctx: PageContext) -> dict[str, Any] | None:
    matched = matched_terms(f"{ctx.page_text} {ctx.form_text} {ctx.form_button_text}", PAYMENT_TERMS)
    if not matched:
        return None
    return make_feature(
        "content.payment_request_language",
        "suspicious",
        "medium",
        {"matched_terms": matched})


@register("content.checkout_shipping_cues")
def feature_content_checkout_shipping_cues(ctx: PageContext) -> dict[str, Any] | None:
    matched = matched_terms(f"{ctx.page_text} {ctx.form_text} {ctx.form_button_text}", CHECKOUT_TERMS)
    if not matched:
        return None
    return make_feature(
        "content.checkout_shipping_cues",
        "neutral",
        "low",
        {
            "matched_terms": matched,
            "number_or_tel_input_count": ctx.number_input_count,
        })


@register("content.transactional_lure")
def feature_content_transactional_lure(ctx: PageContext) -> dict[str, Any] | None:
    matched = matched_terms(ctx.page_text, FILE_LURE_TERMS)
    if not matched:
        return None
    direction = "suspicious" if ctx.credential_form_count else "uncertain"
    return make_feature(
        "content.transactional_lure",
        direction,
        "medium" if direction == "suspicious" else "low",
        {"matched_terms": matched})


@register("content.article_editorial")
def feature_content_article_editorial(ctx: PageContext) -> dict[str, Any] | None:
    matched = matched_terms(f"{ctx.page_text} {ctx.meta_text}", ARTICLE_TERMS)
    if not matched:
        return None
    return make_feature(
        "content.article_editorial",
        "neutral",
        "low",
        {"matched_terms": matched})


@register("content.business_service_catalog")
def feature_content_business_service_catalog(ctx: PageContext) -> dict[str, Any] | None:
    matched = matched_terms(f"{ctx.page_text} {ctx.anchor_text} {ctx.meta_text}", BUSINESS_TERMS)
    if not matched:
        return None
    return make_feature(
        "content.business_service_catalog",
        "neutral",
        "low",
        {"matched_terms": matched})


@register("content.profile_organization")
def feature_content_profile_organization(ctx: PageContext) -> dict[str, Any] | None:
    matched = matched_terms(f"{ctx.page_text} {ctx.anchor_text}", PROFILE_TERMS)
    if not matched:
        return None
    return make_feature(
        "content.profile_organization",
        "neutral",
        "low",
        {"matched_terms": matched})


@register("content.directory_service")
def feature_content_directory_service(ctx: PageContext) -> dict[str, Any] | None:
    matched = matched_terms(f"{ctx.page_text} {ctx.anchor_text}", DIRECTORY_TERMS)
    if not matched:
        return None
    return make_feature(
        "content.directory_service",
        "neutral",
        "low",
        {"matched_terms": matched})


@register("content.storefront")
def feature_content_storefront(ctx: PageContext) -> dict[str, Any] | None:
    store_domains = {domain for domain in ctx.resource_domains if domain and domain_in_set(domain, STORE_PROVIDER_DOMAINS)}
    if not store_domains:
        return None
    return make_feature(
        "content.storefront",
        "neutral",
        "low",
        {"provider_domains": sorted(store_domains)})


@register("chat.widget_presence")
def feature_chat_widget_presence(ctx: PageContext) -> dict[str, Any] | None:
    matched = matched_terms(f"{ctx.page_text} {ctx.anchor_text}", CHAT_TERMS)
    chat_domains = {
        domain
        for domain in ctx.resource_domains + ctx.anchor_domains
        if domain and (domain_in_set(domain, CHAT_PROVIDER_DOMAINS) or known_provider_relationship(domain) == "known_chat_provider")
    }
    for iframe in ctx.iframes:
        domain = absolute_domain(ctx.final_url or ctx.url, norm_text(iframe.get("src")))
        if domain and domain_in_set(domain, CHAT_PROVIDER_DOMAINS):
            chat_domains.add(domain)
    if not matched and not chat_domains:
        return None
    return make_feature(
        "chat.widget_presence",
        "neutral",
        "low",
        {"matched_terms": matched, "provider_domains": sorted(chat_domains)})


@register("content.comment_social_proof")
def feature_content_comment_social_proof(ctx: PageContext) -> dict[str, Any] | None:
    matched = matched_terms(ctx.page_text, COMMENT_TERMS)
    if not matched:
        return None
    return make_feature(
        "content.comment_social_proof",
        "neutral",
        "low",
        {"matched_terms": matched})


@register("meta.robots_noindex")
def feature_meta_robots_noindex(ctx: PageContext) -> dict[str, Any] | None:
    robots_values = ctx.meta_by_name.get("robots", [])
    values = [value for value in robots_values if re.search(r"\b(noindex|nofollow)\b", value, re.I)]
    if not values:
        return None
    return make_feature(
        "meta.robots_noindex",
        "neutral",
        "low",
        {"robots_values": values})


@register("meta.opengraph_mismatch")
def feature_meta_opengraph_mismatch(ctx: PageContext) -> dict[str, Any] | None:
    og_urls = ctx.meta_by_property.get("og:url", [])
    og_images = ctx.meta_by_property.get("og:image", [])
    mismatches: dict[str, Any] = {}
    if og_urls:
        og_domain = registrable_domain(og_urls[0])
        if og_domain and og_domain != ctx.domain:
            mismatches["og_url_domain"] = og_domain
    if og_images:
        og_img_domain = absolute_domain(ctx.final_url or ctx.url, og_images[0])
        if og_img_domain and relationship(ctx.domain, og_img_domain) == "unrelated_third_party":
            mismatches["og_image_domain"] = og_img_domain
    if not mismatches:
        return None
    return make_feature(
        "meta.opengraph_mismatch",
        "neutral",
        "low",
        mismatches)


@register("meta.description_author_consistency")
def feature_meta_description_author_consistency(ctx: PageContext) -> dict[str, Any] | None:
    description = " ".join(ctx.meta_by_name.get("description", []))
    author = " ".join(ctx.meta_by_name.get("author", []))
    text = norm_text(f"{description} {author}")
    if not text:
        return None
    claims = extract_brand_claims(text)
    mismatched = [claim for claim in claims if not brand_matches_domain(claim["brand"], ctx.domain)]
    direction = "suspicious" if mismatched else "neutral"
    return make_feature(
        "meta.description_author_consistency",
        direction,
        "low",
        {"author": author[:200], "brand_claims": claims[:5], "description": description[:300]})


@register("meta.cms_or_security_metadata")
def feature_meta_cms_or_security_metadata(ctx: PageContext) -> dict[str, Any] | None:
    generator = ctx.meta_by_name.get("generator", [])
    security_tokens = []
    for key, values in ctx.meta_by_name.items():
        if "csrf" in key or "verification" in key or "token" in key:
            security_tokens.extend(values)
    if not generator and not security_tokens:
        return None
    return make_feature(
        "meta.cms_or_security_metadata",
        "neutral",
        "low",
        {"generator": generator[:5], "security_tokens": security_tokens[:10]})


@register("resources.external_dependency_profile")
def feature_resources_external_dependency_profile(ctx: PageContext) -> dict[str, Any] | None:
    if not ctx.resource_infos:
        return None
    analytics_domains = {
        item.domain
        for item in ctx.resource_infos
        if item.domain and domain_in_set(item.domain, ANALYTICS_PROVIDER_DOMAINS)
    }
    external = [
        item.domain
        for item in ctx.resource_infos
        if item.domain and item.relationship not in {"same_organization_or_alias", "same_registrable_domain"}
    ]
    if not external:
        return None
    return make_feature(
        "resources.external_dependency_profile",
        "neutral",
        "low",
        {
            "analytics_domains": sorted(analytics_domains),
            "external_resource_count": len(external),
            "resource_count": len(ctx.resource_infos),
        })


@register("resources.brand_asset_consistency")
def feature_resources_brand_asset_consistency(ctx: PageContext) -> dict[str, Any] | None:
    claimed_brands = ctx.brand_names_observed
    brandish_resources = []
    for item in ctx.resource_infos:
        raw_l = lower_text(item.raw)
        if item.kind not in {"favicon_hrefs", "image_src_sample", "stylesheet_href_sample"}:
            continue
        matched_brands = {
            brand
            for brand, aliases in BRAND_ALIASES.items()
            for alias in aliases
            if alias in raw_l
        }
        domain_brand = brand_for_domain(item.domain) if item.domain else None
        if domain_brand:
            matched_brands.add(domain_brand)
        if claimed_brands:
            matched_brands &= claimed_brands
        if matched_brands:
            brandish_resources.append(
                {
                    "brands": sorted(matched_brands),
                    "domain": item.domain,
                    "kind": item.kind,
                    "raw": item.raw[:200],
                }
            )
    if not brandish_resources:
        return None
    direction = "neutral"
    if any(entry["domain"] and relationship(ctx.domain, entry["domain"]) == "unrelated_third_party" for entry in brandish_resources):
        direction = "suspicious"
    return make_feature(
        "resources.brand_asset_consistency",
        direction,
        "low",
        {"brandish_resources": brandish_resources[:10]})


@register("iframe.presence_type")
def feature_iframe_presence_type(ctx: PageContext) -> dict[str, Any] | None:
    if not ctx.iframes:
        return None
    kinds: set[str] = set()
    domains: set[str] = set()
    for iframe in ctx.iframes:
        text = lower_text(" ".join([iframe.get("title") or "", iframe.get("name") or "", iframe.get("src") or ""]))
        domain = absolute_domain(ctx.final_url or ctx.url, norm_text(iframe.get("src")))
        if domain:
            domains.add(domain)
        if any(term in text for term in {"chat", "support"}):
            kinds.add("chat")
        if any(term in text for term in {"map", "maps"}):
            kinds.add("maps")
        if any(term in text for term in {"video", "player", "youtube", "vimeo"}):
            kinds.add("media")
        if any(term in text for term in {"doc", "document", "viewer"}):
            kinds.add("document")
        if any(term in text for term in {"editor", "compose"}):
            kinds.add("editor")
    return make_feature(
        "iframe.presence_type",
        "neutral",
        "low",
        {"domains": sorted(domains), "iframe_count": len(ctx.iframes), "kinds": sorted(kinds)})


@register("iframe.hidden_or_zero_sized")
def feature_iframe_hidden_or_zero_sized(ctx: PageContext) -> dict[str, Any] | None:
    findings = []
    for iframe in ctx.iframes:
        style = lower_text(iframe.get("style"))
        width = lower_text(iframe.get("width"))
        height = lower_text(iframe.get("height"))
        if "display:none" in style.replace(" ", "") or "visibility:hidden" in style.replace(" ", "") or width == "0" or height == "0":
            findings.append({"height": height, "src": norm_text(iframe.get("src"))[:200], "width": width})
    if not findings:
        return None
    return make_feature(
        "iframe.hidden_or_zero_sized",
        "uncertain",
        "low",
        {"findings": findings[:10]})


@register("page.low_semantic_content")
def feature_page_low_semantic_content(ctx: PageContext) -> dict[str, Any] | None:
    if len(ctx.visible_text_l) >= 80 or len(ctx.forms) > 0 or len(ctx.anchor_infos) >= 2:
        return None
    return make_feature(
        "page.low_semantic_content",
        "uncertain",
        "medium",
        {
            "anchor_count": len(ctx.anchor_infos),
            "form_count": len(ctx.forms),
            "visible_text_length": len(ctx.visible_text_l),
        })


@register("page.access_blocked_or_error")
def feature_page_access_blocked_or_error(ctx: PageContext) -> dict[str, Any] | None:
    if ctx.title_l not in ERROR_TITLES and not any(term in ctx.page_text_l for term in {"access denied", "forbidden", "not found", "privacy error"}):
        return None
    return make_feature(
        "page.access_blocked_or_error",
        "uncertain",
        "high",
        {"title": ctx.title})


@register("page.incomplete_render")
def feature_page_incomplete_render(ctx: PageContext) -> dict[str, Any] | None:
    script_count = len(ctx.resources.get("script_src_sample", []))
    prompts = matched_terms(ctx.page_text, SCRIPT_DEPENDENCY_TERMS)
    if not prompts and not (len(ctx.visible_text_l) < 200 and script_count >= 5 and len(ctx.forms) == 0 and len(ctx.anchor_infos) < 3):
        return None
    return make_feature(
        "page.incomplete_render",
        "uncertain",
        "medium",
        {
            "anchor_count": len(ctx.anchor_infos),
            "form_count": len(ctx.forms),
            "matched_terms": prompts,
            "script_count": script_count,
            "visible_text_length": len(ctx.visible_text_l),
        })


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract modular feature families from Stage A JSONL.")
    parser.add_argument("--input", type=Path, default=None, help="Stage A input JSONL.")
    parser.add_argument("--output", type=Path, default=None, help="Output JSONL. Defaults to stdout.")
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bar.")
    parser.add_argument("--list-features", action="store_true", help="List registered feature ids and exit.")
    parser.add_argument(
        "--include",
        type=str,
        default="",
        help="Comma-separated list of feature ids to keep. Empty means all registered features.",
    )
    parser.add_argument(
        "--exclude",
        type=str,
        default="",
        help="Comma-separated list of feature ids to skip.",
    )
    return parser.parse_args()


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


def input_total(path: Path) -> int | None:
    if str(path) == "-":
        return None
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def open_output(path: Path | None):
    if path is None or str(path) == "-":
        return sys.stdout
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.open("w", encoding="utf-8")


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


def selected_registry(include: str, exclude: str) -> list[SelectedFeatureGroup]:
    include_ids = {item.strip() for item in include.split(",") if item.strip()}
    exclude_ids = {item.strip() for item in exclude.split(",") if item.strip()}
    output_ids_by_group = load_group_output_ids()
    selected: list[SelectedFeatureGroup] = []
    for spec in FEATURE_REGISTRY:
        if spec.group_id in exclude_ids:
            continue
        output_ids = output_ids_by_group.get(spec.group_id, (spec.group_id,))
        if include_ids:
            enabled = tuple(output_id for output_id in output_ids if output_id in include_ids)
            if not enabled and spec.group_id in include_ids:
                enabled = output_ids
        else:
            enabled = output_ids
        enabled = tuple(output_id for output_id in enabled if output_id not in exclude_ids)
        if enabled:
            selected.append(
                SelectedFeatureGroup(
                    extractor=spec.extractor,
                    group_id=spec.group_id,
                    output_ids=enabled,
                )
            )
    return selected


def extract_features(record: dict[str, Any], registry: list[SelectedFeatureGroup]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ctx = PageContext(record.get("input") or {})
    features: list[dict[str, Any]] = []
    for spec in registry:
        feature = spec.extractor(ctx)
        if feature is None:
            continue
        base_feature = dict(feature)
        base_feature.pop("supervision", None)
        for output_id in spec.output_ids:
            aliased = dict(base_feature)
            aliased["id"] = output_id
            supervision = FEATURE_SUPERVISION.get(output_id)
            if supervision:
                aliased["supervision"] = supervision
            features.append(aliased)
    return dedupe_features(features), ctx.measurements


def main() -> int:
    args = parse_args()
    if args.list_features:
        seen: set[str] = set()
        for spec in FEATURE_REGISTRY:
            for output_id in load_group_output_ids().get(spec.group_id, (spec.group_id,)):
                if output_id in seen:
                    continue
                seen.add(output_id)
                print(output_id)
        return 0
    if args.input is None:
        raise SystemExit("--input is required unless --list-features is used.")
    registry = selected_registry(args.include, args.exclude)
    output = open_output(args.output)
    close_output = output is not sys.stdout
    total = input_total(args.input)
    written = 0
    try:
        for record in tqdm(
            iter_jsonl(args.input),
            total=total,
            desc="extract_modular_features",
            unit="record",
            disable=args.no_progress,
            file=sys.stderr,
        ):
            features, measurements = extract_features(record, registry)
            out_record = {
                "id": record.get("id"),
                "label": record.get("label"),
                "source": record.get("source"),
                "input": record.get("input") or {},
                "measurements": measurements,
                "features": features,
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
