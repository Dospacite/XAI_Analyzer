from app.model_specs import (
    build_deepseek_user,
    build_llama_user,
    build_qwen_user,
    parse_model_output,
)
from app.service import overall_verdict


DOCUMENT = {"url": "https://example.com/login", "title": "Example Login", "metadata": {}}
FEATURES = {
    "url.scheme_is_https": 1,
    "url.requested_url_length": 25,
    "url.final_url_length": 25,
    "url.hostname_length": 11,
    "url.registrable_domain_length": 11,
    "url.character_entropy": 3.2,
    "html.length": 1000,
    "html.visible_text_length": 100,
    "html.visible_word_count": 20,
    "html.form_count": 1,
    "html.password_input_count": 1,
    "html.credential_form_present": 1,
    "metadata.redirect_count": 0,
}


def test_each_model_gets_its_training_layout():
    assert build_qwen_user(DOCUMENT, FEATURES).startswith("/no_think")
    assert "Selected feature vector:" in build_qwen_user(DOCUMENT, FEATURES)
    assert build_llama_user(DOCUMENT, FEATURES).startswith("Analyze this webpage:")
    deepseek = build_deepseek_user(DOCUMENT, FEATURES)
    assert "### URL ANALYSIS:" in deepseek
    assert "### FORMS & INPUTS:" in deepseek


def test_parse_structured_model_output():
    output = """Phishing risk factors:
- credential.password_input_present

Benign mitigating factors:
- url.https

Reasoning:
The page requests a password and has little content.

Verdict: phishing"""
    parsed = parse_model_output(output)
    assert parsed["verdict"] == "phishing"
    assert parsed["phishing_factors"] == ["credential.password_input_present"]
    assert "little content" in parsed["reasoning"]


def test_parse_deepseek_model_output():
    output = """### PHISHING SIGNALS:
⚠ Password field submits externally

### BENIGN SIGNALS:
✓ HTTPS used

### REASONING:
The external credential flow outweighs HTTPS.

### VERDICT: PHISHING"""
    parsed = parse_model_output(output)
    assert parsed["verdict"] == "phishing"
    assert parsed["phishing_factors"] == ["Password field submits externally"]
    assert parsed["benign_factors"] == ["HTTPS used"]


def test_majority_verdict_ignores_failed_models():
    analyses = [
        {"status": "complete", "verdict": "phishing"},
        {"status": "complete", "verdict": "phishing"},
        {"status": "error", "verdict": "unknown"},
    ]
    assert overall_verdict(analyses) == "phishing"
