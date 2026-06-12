from __future__ import annotations

from datetime import datetime
from typing import Any
from urllib.parse import urlsplit

from fastapi import BackgroundTasks, FastAPI, HTTPException, Response, status
from fastapi.middleware.cors import CORSMiddleware

from .config import get_settings
from .schemas import (
    CreateThreadRequest,
    FollowUpRequest,
    ThreadListResponse,
    ThreadResponse,
    ThreadSummaryResponse,
)
from .service import add_follow_up, fallback_title, new_analyses, process_thread, utcnow
from .store import create_thread as store_create_thread
from .store import delete_thread as store_delete_thread
from .store import get_thread as store_get_thread
from .store import init_store
from .store import list_threads as store_list_threads


app = FastAPI(title="Traceguard API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[get_settings().public_app_url, "http://localhost:5173"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type"],
)


@app.on_event("startup")
async def validate_runtime_settings() -> None:
    if not get_settings().hf_token:
        raise RuntimeError("HF_TOKEN is required")
    await init_store(get_settings().sqlite_path)


def validate_url(raw_url: str) -> str:
    try:
        parsed = urlsplit(raw_url.strip())
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="Invalid URL") from exc
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise HTTPException(status_code=422, detail="Enter a complete HTTP or HTTPS URL")
    if parsed.username or parsed.password:
        raise HTTPException(status_code=422, detail="URLs containing credentials are not allowed")
    return raw_url.strip()


def summary(document: dict[str, Any]) -> ThreadSummaryResponse:
    return ThreadSummaryResponse(
        id=str(document["id"]),
        title=document.get("title") or "Website analysis",
        url=document.get("url") or "",
        status=document.get("status") or "processing",
        overall_verdict=document.get("overall_verdict") or "unknown",
        updated_at=document.get("updated_at") or utcnow(),
    )


def detail(document: dict[str, Any]) -> ThreadResponse:
    source = document.get("source_document") or {}
    metadata = source.get("metadata") or {}
    website = None
    source_json = None
    if source:
        website = {
            "title": source.get("title") or "",
            "final_url": metadata.get("final_url") or source.get("url") or "",
            "status_code": int(metadata.get("status_code") or 0),
            "fetched_at": source.get("fetched_at"),
            "screenshot": source.get("screenshot") or "",
        }
        source_json = {key: value for key, value in source.items() if key != "screenshot"}
    return ThreadResponse(
        **summary(document).model_dump(),
        progress=document.get("progress") or "",
        error=document.get("error"),
        analyses=document.get("analyses") or [],
        messages=document.get("messages") or [],
        website=website,
        document=source_json,
    )


@app.get("/health")
@app.get("/api/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/threads", response_model=ThreadListResponse)
async def list_threads() -> ThreadListResponse:
    items = [summary(document) for document in await store_list_threads()]
    return ThreadListResponse(items=items)


@app.post("/api/threads", response_model=ThreadResponse, status_code=status.HTTP_202_ACCEPTED)
async def create_thread(payload: CreateThreadRequest, background_tasks: BackgroundTasks) -> ThreadResponse:
    url = validate_url(payload.url)
    now = utcnow()
    document = {
        "title": fallback_title(url),
        "url": url,
        "status": "processing",
        "progress": "Queued for isolated retrieval",
        "error": None,
        "overall_verdict": "unknown",
        "source_document": None,
        "features": None,
        "analyses": new_analyses(),
        "messages": [],
        "qwen_history": [],
        "created_at": now,
        "updated_at": now,
    }
    created = await store_create_thread(document)
    background_tasks.add_task(process_thread, created["id"], url)
    return detail(created)


@app.get("/api/threads/{thread_id}", response_model=ThreadResponse)
async def get_thread(thread_id: str) -> ThreadResponse:
    document = await store_get_thread(thread_id)
    if not document:
        raise HTTPException(status_code=404, detail="Thread not found")
    return detail(document)


@app.delete("/api/threads/{thread_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_thread(thread_id: str) -> Response:
    if not await store_delete_thread(thread_id):
        raise HTTPException(status_code=404, detail="Thread not found")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@app.post("/api/threads/{thread_id}/messages", response_model=ThreadResponse)
async def follow_up(thread_id: str, payload: FollowUpRequest) -> ThreadResponse:
    content = payload.content.strip()
    if not content:
        raise HTTPException(status_code=422, detail="Message cannot be empty")
    try:
        document = await add_follow_up(thread_id, content)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Qwen follow-up failed: {type(exc).__name__}") from exc
    return detail(document)
