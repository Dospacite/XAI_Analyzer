from __future__ import annotations

import re
import json
import html as html_lib
from dataclasses import dataclass
from typing import Any, Callable

from bs4 import BeautifulSoup
from .config import Settings

STRUCTURED_SYSTEM = """You are a cybersecurity analyst evaluating webpages for phishing.

You will receive a webpage's URL, hostname, page title, and structured numerical feature data covering URL structure, page layout, forms, links, resources, JavaScript behaviour, and redirects.

Analyze the feature values and output your assessment in this exact format:

Phishing risk factors:
- <feature_id>

Benign mitigating factors:
- <feature_id>

Reasoning:
<1-3 sentences citing the feature values that drove your verdict>

Verdict: phishing

or

Phishing risk factors:
- <feature_id>

Benign mitigating factors:
- <feature_id>

Reasoning:
<1-3 sentences citing the feature values that drove your verdict>

Verdict: benign

Rules:
- List only feature IDs directly supported by the feature values in the prompt.
- If a section has no strong evidence, write `- none`.
- Use `Phishing risk factors:` and `Benign mitigating factors:` as the exact section headers.
- Cite specific feature values in the reasoning.
- The verdict must be exactly `Verdict: phishing` or `Verdict: benign`."""

DEEPSEEK_SYSTEM = """You are CyberGuard, an expert cybersecurity analyst specializing in phishing detection.

You will receive structured web page data including URL characteristics, page structure, forms, links, resources, security indicators, and redirect behavior.

Your task:
1. List PHISHING SIGNALS (⚠) — suspicious indicators found in the data
2. List BENIGN SIGNALS (✓) — legitimate indicators found in the data
3. Write a REASONING paragraph weighing all signals together
4. Give a final VERDICT

Important rules:
- Do NOT base your verdict on a single signal alone
- The absence of navigation or footer alone is NOT sufficient for a phishing verdict
- Always consider URL structure, form behavior, and redirect patterns together
- A page with password fields but strong benign signals may still be legitimate

Output format (strictly follow this):
### PHISHING SIGNALS:
⚠ [signal description]

### BENIGN SIGNALS:
✓ [signal description]

### REASONING:
[2-3 sentence analysis]

### VERDICT: PHISHING or VERDICT: BENIGN"""

GEMMA_SYSTEM = (
    "You are a security classifier for website phishing detection. "
    "Classify websites using only the supplied URL, captured page text, page-structure counts, "
    "and HTTP fetch metadata. Return compact JSON with keys label, confidence, and explanation. "
    "The label must be exactly one of: phishing, legitimate. "
    "Do not use domain registration, WHOIS/RDAP, reputation feeds, or human review."
)


@dataclass(frozen=True)
class ModelSpec:
    key: str
    name: str
    repo_id: str
    endpoint_url: str
    endpoint_name: str
    system: str
    build_user: Callable[[dict[str, Any], dict[str, Any]], str]
    max_new_tokens: int
    temperature: float


def _f(features: dict[str, Any], key: str, default: Any = 0) -> Any:
    value = features.get(key, default)
    return default if value is None else value


def _yes(value: Any) -> str:
    return "yes" if value else "no"


def _pct(value: Any) -> str:
    return f"{float(value or 0):.0%}"


def build_llama_clean_text(document: dict[str, Any], features: dict[str, Any]) -> str:
    url = str(document.get("url") or "")
    title = str(document.get("title") or "")[:200]
    hostname = url.split("//", 1)[1].split("/")[0] if "//" in url else ""
    lines = [line for line in (f"URL: {url}", f"Hostname: {hostname}", f"Page title: {title}") if line.split(": ", 1)[1]]

    flags = []
    if _f(features, "url.host_is_ip_address"):
        flags.append("IP host")
    if _f(features, "url.at_sign_count"):
        flags.append(f"@ sign ({int(_f(features, 'url.at_sign_count'))})")
    if _f(features, "url.https_token_in_hostname"):
        flags.append("'https' in hostname")
    if _f(features, "url.punycode_present"):
        flags.append("punycode")
    if _f(features, "url.non_default_port_present"):
        flags.append("non-default port")
    if _f(features, "url.path_or_query_contains_url"):
        flags.append("URL-in-URL")
    if _f(features, "url.percent_sign_count") > 3:
        flags.append(f"pct-encoded ({int(_f(features, 'url.percent_sign_count'))})")

    lines.extend(
        [
            (
                f"URL patterns: length={int(_f(features, 'url.requested_url_length'))} | "
                f"hostname={int(_f(features, 'url.hostname_length'))} chars "
                f"({int(_f(features, 'url.hostname_digit_count'))} digits) | "
                f"domain_len={int(_f(features, 'url.registrable_domain_length'))} | "
                f"subdomain={int(_f(features, 'url.subdomain_length'))} chars, "
                f"{int(_f(features, 'url.subdomain_label_count'))} labels | "
                f"path={int(_f(features, 'url.path_length'))} chars, "
                f"{int(_f(features, 'url.path_segment_count'))} segments | "
                f"query={int(_f(features, 'url.query_length'))} chars, "
                f"{int(_f(features, 'url.query_parameter_count'))} params | "
                f"entropy={float(_f(features, 'url.character_entropy')):.2f} | "
                f"digit_ratio={float(_f(features, 'url.digit_ratio')):.2f} | "
                f"hyphens={int(_f(features, 'url.hyphen_count'))}"
                + (f" | flags: {', '.join(flags)}" if flags else "")
            ),
            (
                f"Page structure: {int(_f(features, 'html.total_tag_count'))} tags, "
                f"{int(_f(features, 'html.unique_tag_count'))} unique | "
                f"{int(_f(features, 'html.script_tag_count'))} scripts "
                f"({int(_f(features, 'html.external_script_count'))} external, "
                f"{int(_f(features, 'html.inline_script_count'))} inline) | "
                f"{int(_f(features, 'html.image_count'))} images "
                f"({int(_f(features, 'html.external_image_count'))} ext) | "
                f"{int(_f(features, 'html.iframe_count'))} iframes "
                f"({int(_f(features, 'html.external_iframe_count'))} ext) | "
                f"{int(_f(features, 'html.heading_h1_h2_h3_count'))} headings | "
                f"{int(_f(features, 'html.paragraph_count'))} paragraphs"
            ),
            (
                f"Links: {int(_f(features, 'html.anchors_with_href_count'))} with href | "
                f"{int(_f(features, 'html.internal_anchor_count'))} internal, "
                f"{int(_f(features, 'html.external_anchor_count'))} external "
                f"({_pct(_f(features, 'html.external_anchor_ratio'))}) | "
                f"placeholder_ratio={_pct(_f(features, 'html.placeholder_link_ratio'))}"
            ),
        ]
    )
    if _f(features, "html.form_count"):
        lines.append(
            f"Forms: {int(_f(features, 'html.form_count'))} form(s), "
            f"{int(_f(features, 'html.post_form_count'))} POST, "
            f"{int(_f(features, 'html.submit_button_count'))} submit btn | "
            f"{int(_f(features, 'html.input_count'))} inputs "
            f"({int(_f(features, 'html.password_input_count'))} password, "
            f"{int(_f(features, 'html.email_input_count'))} email, "
            f"{int(_f(features, 'html.text_input_count'))} text) | "
            f"{int(_f(features, 'html.hidden_input_count'))} hidden | "
            f"{int(_f(features, 'html.external_form_action_count'))} external action, "
            f"{int(_f(features, 'html.null_form_action_count'))} null action"
            + (" | credential_form" if _f(features, "html.credential_form_present") else "")
            + (" | password->external" if _f(features, "html.password_form_external_action_present") else "")
            + (" | password->null" if _f(features, "html.password_form_null_action_present") else "")
        )
    lines.extend(
        [
            (
                f"Content: {int(_f(features, 'html.visible_text_length'))} chars, "
                f"{int(_f(features, 'html.visible_word_count'))} words | "
                f"text/html={float(_f(features, 'html.visible_text_to_html_ratio')):.1%} | "
                f"nav={_yes(_f(features, 'html.navigation_present'))} | "
                f"footer={_yes(_f(features, 'html.footer_present'))} | "
                f"privacy_link={_yes(_f(features, 'html.privacy_or_terms_link_present'))}"
            ),
            (
                f"Title signals: length={int(_f(features, 'html.title_length'))} | "
                f"domain-in-title={_yes(_f(features, 'html.current_domain_token_in_title'))} | "
                f"registered-domain-in-title={_yes(_f(features, 'html.title_registered_domain_token_present'))} | "
                f"subdomain-in-title={_yes(_f(features, 'html.title_subdomain_token_present'))} | "
                f"title/url overlap={float(_f(features, 'html.title_url_token_overlap_ratio')):.2f}"
            ),
        ]
    )
    js_flags = []
    for key, label in (
        ("html.eval_call_count", "eval"),
        ("html.document_write_count", "document.write"),
        ("html.atob_call_count", "atob"),
    ):
        if _f(features, key):
            js_flags.append(f"{label}({int(_f(features, key))}x)")
    for key, label in (
        ("html.right_click_disabling_present", "right-click disabled"),
        ("html.onmouseover_handler_count", "onmouseover handlers"),
        ("html.javascript_redirect_present", "JS redirect"),
        ("html.alert_or_popup_present", "alert/popup"),
        ("html.hidden_element_present", "hidden elements"),
    ):
        if _f(features, key):
            js_flags.append(label)
    lines.append(f"JS behaviour: {', '.join(js_flags) if js_flags else 'none'}")
    lines.append(
        f"Redirects: {int(_f(features, 'metadata.redirect_count'))} redirect(s), "
        f"{int(_f(features, 'metadata.redirect_domain_change_count'))} domain change(s) | "
        f"scheme_changed: {_yes(_f(features, 'metadata.final_scheme_changed'))} | "
        f"host_changed: {_yes(_f(features, 'metadata.final_host_changed'))}"
    )
    return "\n".join(lines)[:1500]


def build_qwen_user(document: dict[str, Any], features: dict[str, Any]) -> str:
    selected = {
        key: value
        for key, value in features.items()
        if key not in {"url.angle_bracket_count", "url.file_extension"}
    }
    vector = " | ".join(
        f"{key}={float(value):.4g}" if isinstance(value, float) else f"{key}={value}"
        for key, value in sorted(selected.items())
    )
    url = str(document.get("url") or "")
    title = str(document.get("title") or "")
    hostname = url.split("//", 1)[1].split("/")[0] if "//" in url else ""
    clean = (
        f"URL: {url}\nHostname: {hostname}\nPage title: {title}\n"
        f"Selected feature vector: {vector}\n{build_llama_clean_text(document, features)}"
    )
    return f"/no_think\n\nAnalyze this webpage:\n\n{clean[:5000]}"


def build_llama_user(document: dict[str, Any], features: dict[str, Any]) -> str:
    return f"Analyze this webpage:\n\n{build_llama_clean_text(document, features)}"


def build_deepseek_user(document: dict[str, Any], features: dict[str, Any]) -> str:
    url = str(document.get("url") or document.get("metadata", {}).get("final_url") or "N/A")
    title = str(document.get("title") or "")[:100] or "N/A"
    f = features
    feature_text = f"""### URL ANALYSIS:
Total length: {_f(f, 'url.final_url_length')} | Hostname: {_f(f, 'url.hostname_length')} chars | Domain: {_f(f, 'url.registrable_domain_length')} chars
Subdomain: {_f(f, 'url.subdomain_length')} chars | Subdomain labels: {_f(f, 'url.subdomain_label_count')}
Path: {_f(f, 'url.path_length')} chars | Segments: {_f(f, 'url.path_segment_count')} | Query: {_f(f, 'url.query_length')} chars
HTTPS: {'Yes' if _f(f, 'url.scheme_is_https') else 'No'} | IP address: {'Yes' if _f(f, 'url.host_is_ip_address') else 'No'} | Punycode: {'Yes' if _f(f, 'url.punycode_present') else 'No'}
Digits: {float(_f(f, 'url.digit_ratio')) * 100:.1f}% | Entropy: {float(_f(f, 'url.character_entropy')):.2f}
Dots: {_f(f, 'url.dot_count')} | Hyphens: {_f(f, 'url.hyphen_count')} | Special chars: {_f(f, 'url.special_character_count')}
Tokens: {_f(f, 'url.token_count')} | Avg token length: {float(_f(f, 'url.average_token_length')):.1f} | Longest token: {_f(f, 'url.longest_token_length')}

### PAGE STRUCTURE:
HTML length: {int(_f(f, 'html.length')):,} | Visible text: {int(_f(f, 'html.visible_text_length')):,} chars | Words: {_f(f, 'html.visible_word_count')}
Text/HTML ratio: {float(_f(f, 'html.visible_text_to_html_ratio')) * 100:.1f}% | Text entropy: {float(_f(f, 'html.visible_text_entropy')):.2f}
Title length: {_f(f, 'html.title_length')} | Domain token in title: {'Yes' if _f(f, 'html.current_domain_token_in_title') else 'No'}
Total tags: {_f(f, 'html.total_tag_count')} | Unique tags: {_f(f, 'html.unique_tag_count')}
Divs: {_f(f, 'html.div_count')} | Spans: {_f(f, 'html.span_count')} | Paragraphs: {_f(f, 'html.paragraph_count')}
Headings: {_f(f, 'html.heading_h1_h2_h3_count')} | Lists: {_f(f, 'html.list_count')} | Tables: {_f(f, 'html.table_count')}
Footer: {'Yes' if _f(f, 'html.footer_present') else 'No'} | Navigation: {'Yes' if _f(f, 'html.navigation_present') else 'No'} | Privacy/Terms link: {'Yes' if _f(f, 'html.privacy_or_terms_link_present') else 'No'}

### FORMS & INPUTS:
Forms: {_f(f, 'html.form_count')} | POST forms: {_f(f, 'html.post_form_count')}
Inputs: {_f(f, 'html.input_count')} | Text: {_f(f, 'html.text_input_count')} | Password: {_f(f, 'html.password_input_count')} | Email: {_f(f, 'html.email_input_count')}
Hidden inputs: {_f(f, 'html.hidden_input_count')} | Hidden ratio: {float(_f(f, 'html.hidden_input_ratio')) * 100:.1f}%
Submit buttons: {_f(f, 'html.submit_button_count')} | Credential form: {'Yes' if _f(f, 'html.credential_form_present') else 'No'}
Null action forms: {_f(f, 'html.null_form_action_count')} | External action forms: {_f(f, 'html.external_form_action_count')}
Password form null action: {'Yes' if _f(f, 'html.password_form_null_action_present') else 'No'} | Password form external action: {'Yes' if _f(f, 'html.password_form_external_action_present') else 'No'}

### LINKS & ANCHORS:
Anchors: {_f(f, 'html.anchors_with_href_count')} | Internal: {_f(f, 'html.internal_anchor_count')} | External: {_f(f, 'html.external_anchor_count')}
Null/empty anchors: {_f(f, 'html.null_or_empty_anchor_count')} | Placeholder ratio: {float(_f(f, 'html.placeholder_link_ratio')) * 100:.1f}%
External anchor ratio: {float(_f(f, 'html.external_anchor_ratio')) * 100:.1f}%

### RESOURCES:
Images: {_f(f, 'html.image_count')} | External images: {_f(f, 'html.external_image_count')} | External image ratio: {float(_f(f, 'html.external_image_ratio')) * 100:.1f}%
Scripts: {_f(f, 'html.script_tag_count')} | External scripts: {_f(f, 'html.external_script_count')} | Inline scripts: {_f(f, 'html.inline_script_count')}
Stylesheets: {_f(f, 'html.stylesheet_link_count')} | External stylesheets: {_f(f, 'html.external_stylesheet_count')}
Iframes: {_f(f, 'html.iframe_count')} | External iframes: {_f(f, 'html.external_iframe_count')}
Favicons: {_f(f, 'html.favicon_count')} | Total resource URLs: {_f(f, 'html.resource_url_count')}
Unique external domains: {_f(f, 'html.unique_external_resource_domain_count')} | External resource ratio: {float(_f(f, 'html.external_resource_ratio')) * 100:.1f}%

### SECURITY INDICATORS:
Hidden element: {'Yes' if _f(f, 'html.hidden_element_present') else 'No'} | JS redirect: {'Yes' if _f(f, 'html.javascript_redirect_present') else 'No'}
eval() calls: {_f(f, 'html.eval_call_count')} | atob() calls: {_f(f, 'html.atob_call_count')} | document.write: {_f(f, 'html.document_write_count')}
Right-click disabled: {'Yes' if _f(f, 'html.right_click_disabling_present') else 'No'} | Alert/popup: {'Yes' if _f(f, 'html.alert_or_popup_present') else 'No'}
Meta refresh: {_f(f, 'html.meta_refresh_count')} | Meta tags: {_f(f, 'html.meta_tag_count')}

### REDIRECTS:
Redirect count: {_f(f, 'metadata.redirect_count')} | Domain changes: {_f(f, 'metadata.redirect_domain_change_count')}
URL changed: {'Yes' if _f(f, 'metadata.final_url_changed') else 'No'} | Host changed: {'Yes' if _f(f, 'metadata.final_host_changed') else 'No'} | Scheme changed: {'Yes' if _f(f, 'metadata.final_scheme_changed') else 'No'}"""
    return f"### TARGET:\nURL: {url}\nPage Title: {title}\n\n{feature_text}"


def _compact_text(text: str, max_chars: int) -> str:
    value = re.sub(r"\s+", " ", html_lib.unescape(text or "")).strip()
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 20].rstrip() + " ... [truncated]"


def _gemma_visible_text(html: str, max_chars: int = 6000) -> str:
    soup = BeautifulSoup(html or "", "lxml")
    for tag in soup(["script", "style", "noscript", "svg", "canvas"]):
        tag.decompose()
    return _compact_text(soup.get_text(" "), max_chars)


def build_gemma_user(document: dict[str, Any], features: dict[str, Any]) -> str:
    metadata = document.get("metadata") if isinstance(document.get("metadata"), dict) else {}
    requested_url = str(document.get("url") or "")
    final_url = str(metadata.get("final_url") or metadata.get("url") or requested_url)
    html = str(document.get("html") or "")
    title = _compact_text(str(document.get("title") or ""), 180) or "No title captured."
    text = _gemma_visible_text(html) or "No page body text captured."
    stats = {
        "text_chars": len(text),
        "links_or_form_targets": int(_f(features, "html.anchors_with_href_count"))
        + int(_f(features, "html.form_count")),
        "script_link_iframe_resources": int(_f(features, "html.script_tag_count"))
        + int(_f(features, "html.stylesheet_link_count"))
        + int(_f(features, "html.iframe_count")),
        "forms": int(_f(features, "html.form_count")),
        "password_fields": int(_f(features, "html.password_input_count")),
        "input_fields": int(_f(features, "html.input_count")),
        "iframes": int(_f(features, "html.iframe_count")),
        "scripts": int(_f(features, "html.script_tag_count")),
        "status_code": metadata.get("status_code"),
        "redirect_count": int(_f(features, "metadata.redirect_count")),
    }
    page_stats = "\n".join(f"- {key}: {value}" for key, value in stats.items())
    input_features = "\n".join(
        [
            "# Information:",
            "## URL:",
            final_url,
            "## Title:",
            title,
            "## Content:",
            text,
            "## Page Structure:",
            page_stats,
        ]
    )
    return f"Classify the following website evidence using only the fields shown below.\n\n{input_features}\n# Pred:"


def model_specs(settings: Settings) -> list[ModelSpec]:
    return [
        ModelSpec(
            key="qwen",
            name="Qwen3 4B Explainable",
            repo_id=settings.hf_qwen_repo_id,
            endpoint_url=settings.hf_qwen_endpoint_url,
            endpoint_name=settings.hf_qwen_endpoint_name,
            system=STRUCTURED_SYSTEM,
            build_user=build_qwen_user,
            max_new_tokens=220,
            temperature=0,
        ),
        ModelSpec(
            key="llama",
            name="Llama 3.1 8B",
            repo_id=settings.hf_llama_repo_id,
            endpoint_url=settings.hf_llama_endpoint_url,
            endpoint_name=settings.hf_llama_endpoint_name,
            system=STRUCTURED_SYSTEM,
            build_user=build_llama_user,
            max_new_tokens=256,
            temperature=0,
        ),
        ModelSpec(
            key="deepseek",
            name="DeepSeek R1 Qwen 7B",
            repo_id=settings.hf_deepseek_repo_id,
            endpoint_url=settings.hf_deepseek_endpoint_url,
            endpoint_name=settings.hf_deepseek_endpoint_name,
            system=DEEPSEEK_SYSTEM,
            build_user=build_deepseek_user,
            max_new_tokens=320,
            temperature=0.1,
        ),
        ModelSpec(
            key="gemma",
            name="Gemma 4 E4B Unsloth",
            repo_id=settings.hf_gemma_repo_id,
            endpoint_url=settings.hf_gemma_endpoint_url,
            endpoint_name=settings.hf_gemma_endpoint_name,
            system=GEMMA_SYSTEM,
            build_user=build_gemma_user,
            max_new_tokens=256,
            temperature=0,
        ),
    ]


def parse_model_output(text: str) -> dict[str, Any]:
    cleaned = re.sub(r"<think>.*?</think>", "", text or "", flags=re.IGNORECASE | re.DOTALL).strip()
    json_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if json_match:
        try:
            payload = json.loads(json_match.group(0))
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict):
            label = str(payload.get("label") or payload.get("verdict") or "").strip().lower()
            verdict = "phishing" if label == "phishing" else "benign" if label in {"legitimate", "benign"} else "unknown"
            explanation = str(payload.get("explanation") or payload.get("reasoning") or "").strip()
            confidence = payload.get("confidence")
            confidence_text = f" Confidence: {confidence}." if confidence is not None else ""
            factor_lines = [
                line.strip()
                for line in explanation.splitlines()
                if line.strip() and not line.lower().startswith("predicted as")
            ]
            return {
                "verdict": verdict,
                "phishing_factors": factor_lines if verdict == "phishing" else [],
                "benign_factors": factor_lines if verdict == "benign" else [],
                "reasoning": (explanation + confidence_text).strip(),
                "raw_output": cleaned,
            }
    verdicts = re.findall(r"\bverdict\s*[:\-]\s*\**\s*(phishing|benign)\b", cleaned, re.IGNORECASE)
    verdict = verdicts[-1].lower() if verdicts else "unknown"

    def section(start_patterns: list[str], end_patterns: list[str]) -> list[str]:
        start = "|".join(start_patterns)
        end = "|".join(end_patterns)
        match = re.search(
            rf"(?:{start})\s*:?\s*(.*?)(?=(?:{end})\s*:|\Z)",
            cleaned,
            re.IGNORECASE | re.DOTALL,
        )
        if not match:
            return []
        lines = []
        for raw in match.group(1).splitlines():
            item = re.sub(r"^\s*(?:[-*]|⚠|✓|—)\s*", "", raw).strip()
            if item and item.lower() not in {"none", "none detected"}:
                lines.append(item)
        return lines

    phishing = section(
        [r"#{0,3}\s*phishing risk factors", r"#{0,3}\s*phishing signals"],
        [r"#{0,3}\s*benign mitigating factors", r"#{0,3}\s*benign signals", r"#{0,3}\s*reasoning", r"#{0,3}\s*verdict"],
    )
    benign = section(
        [r"#{0,3}\s*benign mitigating factors", r"#{0,3}\s*benign signals"],
        [r"#{0,3}\s*reasoning", r"#{0,3}\s*verdict"],
    )
    reasoning_lines = section(
        [r"#{0,3}\s*reasoning"],
        [r"#{0,3}\s*verdict"],
    )
    return {
        "verdict": verdict,
        "phishing_factors": phishing,
        "benign_factors": benign,
        "reasoning": " ".join(reasoning_lines).strip(),
        "raw_output": cleaned,
    }
