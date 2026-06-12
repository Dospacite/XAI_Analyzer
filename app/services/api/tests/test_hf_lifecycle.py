import httpx
import pytest

from app import clients
from app.clients import run_hf_model
from app.config import Settings
from app.model_specs import ModelSpec


def _spec() -> ModelSpec:
    return ModelSpec(
        key="qwen",
        name="Qwen",
        repo_id="Dospacite/test-merged",
        endpoint_url="https://model.example",
        endpoint_name="traceguard-test",
        system="System",
        build_user=lambda document, features: "User",
        max_new_tokens=8,
        temperature=0,
    )


@pytest.mark.asyncio
async def test_hf_endpoint_resumes_runs_and_pauses(monkeypatch):
    clients._endpoint_locks.clear()
    clients._endpoint_active_uses.clear()
    calls: list[str] = []
    status_reads = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal status_reads
        if request.url.host == "api.endpoints.huggingface.cloud":
            if request.url.path.endswith("/resume"):
                calls.append("resume")
                return httpx.Response(200, json={"status": {"state": "initializing"}})
            if request.url.path.endswith("/pause"):
                calls.append("pause")
                return httpx.Response(200, json={"status": {"state": "paused"}})
            status_reads += 1
            calls.append("status")
            state = "running" if status_reads > 1 else "initializing"
            return httpx.Response(200, json={"status": {"state": state}})

        calls.append("inference")
        return httpx.Response(200, json={"generated_text": "Verdict: benign"})

    transport = httpx.MockTransport(handler)

    class MockAsyncClient(httpx.AsyncClient):
        def __init__(self, *args, **kwargs):
            super().__init__(transport=transport, timeout=kwargs.get("timeout"))

    monkeypatch.setattr("app.clients.httpx.AsyncClient", MockAsyncClient)

    settings = Settings(
        hf_token="token",
        hf_manage_endpoint_lifecycle=True,
        hf_endpoint_namespace="Dospacite",
        hf_endpoint_poll_seconds=0,
        hf_endpoint_start_timeout_seconds=2,
    )
    output = await run_hf_model(settings, _spec(), {}, {})

    assert output == "Verdict: benign"
    assert calls == ["resume", "status", "status", "inference", "pause"]


@pytest.mark.asyncio
async def test_hf_endpoint_lifecycle_can_be_disabled(monkeypatch):
    clients._endpoint_locks.clear()
    clients._endpoint_active_uses.clear()
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url.host))
        return httpx.Response(200, json={"generated_text": "Verdict: benign"})

    transport = httpx.MockTransport(handler)

    class MockAsyncClient(httpx.AsyncClient):
        def __init__(self, *args, **kwargs):
            super().__init__(transport=transport, timeout=kwargs.get("timeout"))

    monkeypatch.setattr("app.clients.httpx.AsyncClient", MockAsyncClient)

    settings = Settings(hf_token="token", hf_manage_endpoint_lifecycle=False)
    output = await run_hf_model(settings, _spec(), {}, {})

    assert output == "Verdict: benign"
    assert calls == ["model.example"]


@pytest.mark.asyncio
async def test_hf_endpoint_refuses_to_run_without_token(monkeypatch):
    clients._endpoint_locks.clear()
    clients._endpoint_active_uses.clear()

    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("endpoint should not be called without HF_TOKEN")

    transport = httpx.MockTransport(handler)

    class MockAsyncClient(httpx.AsyncClient):
        def __init__(self, *args, **kwargs):
            super().__init__(transport=transport, timeout=kwargs.get("timeout"))

    monkeypatch.setattr("app.clients.httpx.AsyncClient", MockAsyncClient)

    settings = Settings(hf_token="", hf_manage_endpoint_lifecycle=False)
    with pytest.raises(clients.UpstreamError, match="HF_TOKEN is required"):
        await run_hf_model(settings, _spec(), {}, {})
