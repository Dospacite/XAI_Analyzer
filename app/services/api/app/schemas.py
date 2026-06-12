from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

Verdict = Literal["phishing", "benign", "unknown"]
ThreadStatus = Literal["processing", "ready", "error"]


class CreateThreadRequest(BaseModel):
    url: str = Field(min_length=8, max_length=2_048)


class FollowUpRequest(BaseModel):
    content: str = Field(min_length=1, max_length=4_000)


class AnalysisResponse(BaseModel):
    key: str
    name: str
    status: Literal["pending", "running", "complete", "error"]
    verdict: Verdict = "unknown"
    phishing_factors: list[str] = Field(default_factory=list)
    benign_factors: list[str] = Field(default_factory=list)
    reasoning: str = ""
    raw_output: str = ""
    error: str | None = None


class MessageResponse(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    created_at: datetime


class WebsiteResponse(BaseModel):
    title: str = ""
    final_url: str = ""
    status_code: int = 0
    fetched_at: datetime | None = None
    screenshot: str = ""


class ThreadSummaryResponse(BaseModel):
    id: str
    title: str
    url: str
    status: ThreadStatus
    overall_verdict: Verdict
    updated_at: datetime


class ThreadResponse(ThreadSummaryResponse):
    progress: str = ""
    error: str | None = None
    analyses: list[AnalysisResponse]
    messages: list[MessageResponse]
    website: WebsiteResponse | None = None
    document: dict[str, Any] | None = None


class ThreadListResponse(BaseModel):
    items: list[ThreadSummaryResponse]
