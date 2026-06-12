from __future__ import annotations

import asyncio
import logging
from typing import Any
from urllib.parse import quote

import httpx

from .config import Settings
from .model_specs import ModelSpec


class UpstreamError(RuntimeError):
    pass


HF_ENDPOINTS_API = "https://api.endpoints.huggingface.cloud/v2"
logger = logging.getLogger(__name__)
_endpoint_locks: dict[str, asyncio.Lock] = {}
_endpoint_active_uses: dict[str, int] = {}


def _message_from_response(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.text[:300] or f"HTTP {response.status_code}"
    if isinstance(payload, dict):
        detail = payload.get("error") or payload.get("detail") or payload.get("message")
        if isinstance(detail, dict):
            return str(detail.get("message") or detail)[:300]
        if detail:
            return str(detail)[:300]
    return f"HTTP {response.status_code}"


def _endpoint_api_url(settings: Settings, endpoint_name: str, suffix: str = "") -> str:
    namespace = quote(settings.hf_endpoint_namespace, safe="")
    name = quote(endpoint_name, safe="")
    return f"{HF_ENDPOINTS_API}/endpoint/{namespace}/{name}{suffix}"


def _endpoint_headers(settings: Settings) -> dict[str, str]:
    if not settings.hf_token:
        raise UpstreamError("HF_TOKEN is required when HF_MANAGE_ENDPOINT_LIFECYCLE=true")
    return {"Authorization": f"Bearer {settings.hf_token}"}


def _hf_inference_headers(settings: Settings) -> dict[str, str]:
    if not settings.hf_token:
        raise UpstreamError("HF_TOKEN is required to call Hugging Face endpoints")
    return {"Authorization": f"Bearer {settings.hf_token}"}


def _endpoint_key(settings: Settings, endpoint_name: str) -> str:
    return f"{settings.hf_endpoint_namespace}/{endpoint_name}"


def _endpoint_lock(key: str) -> asyncio.Lock:
    lock = _endpoint_locks.get(key)
    if lock is None:
        lock = asyncio.Lock()
        _endpoint_locks[key] = lock
    return lock


async def _read_endpoint_state(
    client: httpx.AsyncClient,
    settings: Settings,
    endpoint_name: str,
) -> dict[str, Any]:
    response = await client.get(
        _endpoint_api_url(settings, endpoint_name),
        headers=_endpoint_headers(settings),
    )
    if response.is_error:
        raise UpstreamError(f"Could not read endpoint {endpoint_name}: {_message_from_response(response)}")
    payload = response.json()
    status = payload.get("status") if isinstance(payload, dict) else None
    return status if isinstance(status, dict) else {}


async def _resume_endpoint(
    client: httpx.AsyncClient,
    settings: Settings,
    endpoint_name: str,
) -> None:
    response = await client.post(
        _endpoint_api_url(settings, endpoint_name, "/resume"),
        headers=_endpoint_headers(settings),
    )
    if response.is_error and "already running" not in response.text.lower():
        raise UpstreamError(f"Could not resume endpoint {endpoint_name}: {_message_from_response(response)}")


async def _pause_endpoint(
    client: httpx.AsyncClient,
    settings: Settings,
    endpoint_name: str,
) -> None:
    response = await client.post(
        _endpoint_api_url(settings, endpoint_name, "/pause"),
        headers=_endpoint_headers(settings),
    )
    if response.is_error and "already paused" not in response.text.lower():
        raise UpstreamError(f"Could not pause endpoint {endpoint_name}: {_message_from_response(response)}")


async def _wait_for_endpoint_ready(
    client: httpx.AsyncClient,
    settings: Settings,
    endpoint_name: str,
) -> None:
    deadline = asyncio.get_running_loop().time() + settings.hf_endpoint_start_timeout_seconds
    while True:
        status = await _read_endpoint_state(client, settings, endpoint_name)
        state = str(status.get("state") or "").lower()
        if state == "running":
            return
        if state == "failed":
            message = status.get("errorMessage") or status.get("message") or "Endpoint failed to start"
            raise UpstreamError(f"Endpoint {endpoint_name} failed to start: {message}")
        if asyncio.get_running_loop().time() >= deadline:
            raise UpstreamError(f"Endpoint {endpoint_name} did not become ready before timeout")
        await asyncio.sleep(settings.hf_endpoint_poll_seconds)


async def _ensure_endpoint_ready(
    client: httpx.AsyncClient,
    settings: Settings,
    spec: ModelSpec,
) -> None:
    if not settings.hf_manage_endpoint_lifecycle or not spec.endpoint_name:
        return
    await _resume_endpoint(client, settings, spec.endpoint_name)
    await _wait_for_endpoint_ready(client, settings, spec.endpoint_name)


async def _acquire_endpoint_use(
    client: httpx.AsyncClient,
    settings: Settings,
    spec: ModelSpec,
) -> str | None:
    if not settings.hf_manage_endpoint_lifecycle or not spec.endpoint_name:
        return None

    key = _endpoint_key(settings, spec.endpoint_name)
    async with _endpoint_lock(key):
        if _endpoint_active_uses.get(key, 0) == 0:
            await _ensure_endpoint_ready(client, settings, spec)
        _endpoint_active_uses[key] = _endpoint_active_uses.get(key, 0) + 1
    return key


async def _release_endpoint_use(
    client: httpx.AsyncClient,
    settings: Settings,
    spec: ModelSpec,
    key: str | None,
) -> None:
    if key is None or not spec.endpoint_name:
        return

    async with _endpoint_lock(key):
        remaining = max(_endpoint_active_uses.get(key, 1) - 1, 0)
        if remaining:
            _endpoint_active_uses[key] = remaining
            return
        _endpoint_active_uses.pop(key, None)
        await _pause_endpoint(client, settings, spec.endpoint_name)


async def scrape_website(settings: Settings, url: str) -> dict[str, Any]:
    timeout = httpx.Timeout(connect=10, read=45, write=10, pool=10)
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(
            f"{settings.scraper_url.rstrip('/')}/scrape",
            headers={"X-Scraper-Token": settings.scraper_token},
            json={"url": url},
        )
    if response.is_error:
        raise UpstreamError(_message_from_response(response))
    return response.json()


def _extract_generated_text(payload: Any) -> str:
    if isinstance(payload, str):
        return payload
    if isinstance(payload, dict):
        for key in ("generated_text", "text", "output"):
            if isinstance(payload.get(key), str):
                return payload[key]
        if isinstance(payload.get("choices"), list) and payload["choices"]:
            choice = payload["choices"][0]
            if isinstance(choice, dict):
                message = choice.get("message")
                if isinstance(message, dict) and isinstance(message.get("content"), str):
                    return message["content"]
                if isinstance(choice.get("text"), str):
                    return choice["text"]
    if isinstance(payload, list) and payload:
        return _extract_generated_text(payload[0])
    raise UpstreamError("The endpoint response did not contain generated text")


def _qwen_style_chat_prompt(system: str, user: str, *, disable_thinking: bool = False) -> str:
    prompt = (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    if disable_thinking:
        prompt += "<think>\n\n</think>\n\n"
    return prompt


def _deepseek_style_chat_prompt(system: str, user: str) -> str:
    return f"<｜begin▁of▁sentence｜>{system}<｜User｜>{user}<｜Assistant｜>"


def _llama_style_chat_prompt(system: str, user: str) -> str:
    return (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n\n"
        f"{system}<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"{user}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )


async def run_hf_model(
    settings: Settings,
    spec: ModelSpec,
    document: dict[str, Any],
    features: dict[str, Any],
) -> str:
    if not spec.endpoint_url:
        raise UpstreamError(
            f"Dedicated endpoint not configured for {spec.repo_id}. "
            f"Set HF_{spec.key.upper()}_ENDPOINT_URL after deploying the Hugging Face repository."
        )
    if not settings.hf_token:
        raise UpstreamError("HF_TOKEN is required to call Hugging Face endpoints")

    user_prompt = spec.build_user(document, features)
    parameters = {
        "max_new_tokens": spec.max_new_tokens,
        "temperature": spec.temperature,
        "do_sample": spec.temperature > 0,
        "repetition_penalty": 1.1,
        "return_full_text": False,
    }
    if spec.repo_id.endswith("-merged"):
        if spec.key == "deepseek":
            prompt = _deepseek_style_chat_prompt(spec.system, user_prompt)
            payload = {
                "inputs": prompt,
                "parameters": parameters,
            }
        elif spec.key == "llama":
            prompt = _llama_style_chat_prompt(spec.system, user_prompt)
            payload = {
                "inputs": prompt,
                "parameters": parameters,
            }
        elif spec.key == "gemma":
            payload = {
                "inputs": {
                    "messages": [
                        {"role": "system", "content": spec.system},
                        {"role": "user", "content": user_prompt},
                    ]
                },
                "parameters": parameters,
            }
        else:
            prompt = _qwen_style_chat_prompt(
                spec.system,
                user_prompt,
                disable_thinking=spec.key == "qwen",
            )
            payload = {
                "inputs": prompt,
                "parameters": parameters,
            }
    else:
        payload = {
            "inputs": {
                "messages": [
                    {"role": "system", "content": spec.system},
                    {"role": "user", "content": user_prompt},
                ]
            },
            "parameters": parameters,
    }
    timeout = httpx.Timeout(connect=20, read=settings.request_timeout_seconds, write=30, pool=20)
    async with httpx.AsyncClient(timeout=timeout) as client:
        endpoint_use_key: str | None = None
        response: httpx.Response | None = None
        try:
            endpoint_use_key = await _acquire_endpoint_use(client, settings, spec)
            for attempt in range(7):
                response = await client.post(
                    spec.endpoint_url,
                    headers=_hf_inference_headers(settings),
                    json=payload,
                )
                if response.status_code not in {502, 503, 504}:
                    break
                if attempt < 6:
                    await asyncio.sleep(10)
            assert response is not None
            if response.is_error:
                raise UpstreamError(_message_from_response(response))
            return _extract_generated_text(response.json())
        finally:
            if endpoint_use_key is not None:
                try:
                    await _release_endpoint_use(client, settings, spec, endpoint_use_key)
                except UpstreamError:
                    logger.exception("Could not pause Hugging Face endpoint %s", spec.endpoint_name)


def _qwen_url(base_url: str) -> str:
    clean = base_url.rstrip("/")
    return clean if clean.endswith("/chat/completions") else f"{clean}/chat/completions"


async def qwen_chat(
    settings: Settings,
    messages: list[dict[str, str]],
    *,
    max_tokens: int = 900,
    temperature: float = 0.2,
) -> str:
    if not settings.qwen_base_url or not settings.qwen_api_key:
        raise UpstreamError("Qwen API credentials are not configured")
    timeout = httpx.Timeout(connect=15, read=settings.request_timeout_seconds, write=20, pool=15)
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(
            _qwen_url(settings.qwen_base_url),
            headers={
                "Authorization": f"Bearer {settings.qwen_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": settings.qwen_model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": False,
            },
        )
    if response.is_error:
        raise UpstreamError(_message_from_response(response))
    return _extract_generated_text(response.json()).strip()


async def generate_thread_title(settings: Settings, url: str) -> str:
    try:
        answer = await asyncio.wait_for(
            qwen_chat(
                settings,
                [
                    {
                        "role": "system",
                        "content": "Name website analysis threads. Return only a concise title of at most three words.",
                    },
                    {
                        "role": "user",
                        "content": f"Create a maximum three-word thread name for this URL: {url}",
                    },
                ],
                max_tokens=20,
                temperature=0.1,
            ),
            timeout=12,
        )
        words = answer.replace('"', "").replace("'", "").strip().split()
        if words:
            return " ".join(words[:3])[:60]
    except Exception:
        pass

    hostname = httpx.URL(url).host or "Website analysis"
    label = hostname.removeprefix("www.").split(".")[0].replace("-", " ").strip()
    return " ".join(label.title().split()[:3]) or "Website analysis"
