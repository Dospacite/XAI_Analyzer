from __future__ import annotations

import asyncio
import base64
import glob
import ipaddress
import socket
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlsplit

from fastapi import Depends, FastAPI, Header, HTTPException, status
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from scrapling.fetchers import DynamicFetcher, Fetcher


class Settings(BaseSettings):
    scraper_token: str = "local-development-token"
    scraper_max_bytes: int = 5_000_000
    scraper_timeout_seconds: float = 30

    model_config = SettingsConfigDict(env_file=None, extra="ignore")


settings = Settings()
app = FastAPI(
    title="Traceguard isolated scraper",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)


class ScrapeRequest(BaseModel):
    url: str = Field(min_length=8, max_length=2_048)


class ScrapeResponse(BaseModel):
    url: str
    title: str
    html: str
    fetched_at: datetime
    error: None = None
    metadata: dict[str, Any]
    screenshot: str = ""


def require_token(x_scraper_token: str = Header(default="")) -> None:
    if not settings.scraper_token or x_scraper_token != settings.scraper_token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")


def validate_public_url(raw_url: str) -> str:
    try:
        parsed = urlsplit(raw_url)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="Invalid URL") from exc

    if parsed.scheme not in {"http", "https"}:
        raise HTTPException(status_code=422, detail="Only HTTP and HTTPS URLs are allowed")
    if not parsed.hostname or parsed.username or parsed.password:
        raise HTTPException(status_code=422, detail="URL host is invalid")
    if parsed.port and parsed.port not in {80, 443}:
        raise HTTPException(status_code=422, detail="Only ports 80 and 443 are allowed")

    hostname = parsed.hostname.rstrip(".").lower()
    if hostname in {"localhost", "localhost.localdomain"} or hostname.endswith(".localhost"):
        raise HTTPException(status_code=422, detail="Local targets are not allowed")

    try:
        addresses = {
            item[4][0]
            for item in socket.getaddrinfo(
                hostname,
                parsed.port or (443 if parsed.scheme == "https" else 80),
                type=socket.SOCK_STREAM,
            )
        }
    except socket.gaierror as exc:
        raise HTTPException(status_code=422, detail="The hostname could not be resolved") from exc

    if not addresses:
        raise HTTPException(status_code=422, detail="The hostname did not resolve")
    for address in addresses:
        ip = ipaddress.ip_address(address)
        if not ip.is_global:
            raise HTTPException(status_code=422, detail="Private or reserved network targets are not allowed")

    return raw_url


def _headers_dict(response: Any) -> dict[str, str]:
    headers = getattr(response, "headers", {}) or {}
    try:
        return {str(key): str(value) for key, value in dict(headers).items()}
    except (TypeError, ValueError):
        return {}


def _response_body(response: Any) -> bytes:
    body = getattr(response, "body", b"")
    if isinstance(body, str):
        return body.encode("utf-8", errors="replace")
    return bytes(body or b"")


def _redirect_history(response: Any) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for item in list(getattr(response, "history", []) or [])[:8]:
        output.append(
            {
                "url": str(getattr(item, "url", "")),
                "status_code": int(getattr(item, "status", 0) or 0),
            }
        )
    return output


def _fetch_static(url: str) -> Any:
    return Fetcher.get(
        url,
        impersonate="chrome",
        stealthy_headers=True,
        follow_redirects="safe",
        max_redirects=8,
        timeout=settings.scraper_timeout_seconds,
        retries=1,
        retry_delay=0.5,
        verify=True,
    )


def _fetch_dynamic(url: str, screenshot: dict[str, str]) -> Any:
    def capture(page: Any) -> None:
        image = page.screenshot(full_page=False, type="png")
        screenshot["data_url"] = "data:image/png;base64," + base64.b64encode(image).decode("ascii")

    executables = glob.glob("/ms-playwright/chromium_headless_shell-*/chrome-headless-shell-linux64/chrome-headless-shell")
    executable_path = executables[0] if executables else None

    return DynamicFetcher.fetch(
        url,
        headless=True,
        wait=5_000,
        network_idle=True,
        load_dom=True,
        timeout=settings.scraper_timeout_seconds * 1000,
        retries=1,
        retry_delay=0.5,
        page_action=capture,
        extra_flags=[
            "--disable-dev-shm-usage",
            "--no-sandbox",
            "--disable-setuid-sandbox",
            "--disable-crash-reporter",
            "--disable-crashpad",
        ],
        additional_args={"viewport": {"width": 1365, "height": 768}},
        executable_path=executable_path,
    )


def fetch_document(url: str) -> ScrapeResponse:
    started = datetime.now(timezone.utc)
    screenshot: dict[str, str] = {}
    try:
        response = _fetch_dynamic(url, screenshot)
    except Exception:
        response = _fetch_static(url)
    body = _response_body(response)
    headers = _headers_dict(response)

    declared_length = int(headers.get("content-length", headers.get("Content-Length", "0")) or 0)
    if declared_length > settings.scraper_max_bytes or len(body) > settings.scraper_max_bytes:
        raise HTTPException(status_code=413, detail="The response is larger than the configured limit")

    encoding = str(getattr(response, "encoding", "") or "utf-8")
    try:
        html = body.decode(encoding, errors="replace")
    except LookupError:
        encoding = "utf-8"
        html = body.decode("utf-8", errors="replace")

    final_url = str(getattr(response, "url", "") or url)
    validate_public_url(final_url)
    title = ""
    try:
        title = str(response.css("title::text").get() or "").strip()[:300]
    except Exception:
        title = ""

    history = _redirect_history(response)
    return ScrapeResponse(
        url=url,
        title=title,
        html=html,
        fetched_at=datetime.now(timezone.utc),
        metadata={
            "url": url,
            "status_code": int(getattr(response, "status", 0) or 0),
            "headers": headers,
            "encoding": encoding,
            "elapsed_ms": round((datetime.now(timezone.utc) - started).total_seconds() * 1000, 3),
            "final_url": final_url,
            "redirect_count": len(history),
            "redirect_history": history,
            "content_length": len(body),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "screenshot_captured": bool(screenshot.get("data_url")),
        },
        screenshot=screenshot.get("data_url", ""),
    )


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/scrape", response_model=ScrapeResponse, dependencies=[Depends(require_token)])
async def scrape(payload: ScrapeRequest) -> ScrapeResponse:
    url = await asyncio.to_thread(validate_public_url, payload.url)
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(fetch_document, url),
            timeout=settings.scraper_timeout_seconds + 5,
        )
    except HTTPException:
        raise
    except TimeoutError as exc:
        raise HTTPException(status_code=504, detail="The target website timed out") from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Website retrieval failed: {type(exc).__name__}") from exc
