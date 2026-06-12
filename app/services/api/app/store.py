from __future__ import annotations

import asyncio
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

_lock = asyncio.Lock()
_sqlite_path = "traceguard.sqlite3"


def _connect() -> sqlite3.Connection:
    connection = sqlite3.connect(_sqlite_path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    return connection


def _encode(value: Any) -> Any:
    if isinstance(value, datetime):
        return {"__type": "datetime", "value": value.isoformat()}
    if isinstance(value, dict):
        return {key: _encode(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_encode(item) for item in value]
    return value


def _decode(value: Any) -> Any:
    if isinstance(value, dict):
        if value.get("__type") == "datetime" and isinstance(value.get("value"), str):
            return datetime.fromisoformat(value["value"])
        return {key: _decode(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode(item) for item in value]
    return value


def _dump(document: dict[str, Any]) -> str:
    return json.dumps(_encode(document), separators=(",", ":"))


def _load(raw: str) -> dict[str, Any]:
    return _decode(json.loads(raw))


async def init_store(sqlite_path: str) -> None:
    global _sqlite_path
    _sqlite_path = sqlite_path
    path = Path(sqlite_path)
    if path.parent != Path("."):
        path.parent.mkdir(parents=True, exist_ok=True)
    async with _lock:
        with _connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS threads (
                    id TEXT PRIMARY KEY,
                    updated_at TEXT NOT NULL,
                    document TEXT NOT NULL
                )
                """
            )
            connection.execute("CREATE INDEX IF NOT EXISTS idx_threads_updated_at ON threads(updated_at DESC)")


def _set_path(document: dict[str, Any], path: str, value: Any) -> None:
    target: Any = document
    parts = path.split(".")
    for part in parts[:-1]:
        target = target[int(part)] if isinstance(target, list) else target[part]
    last = parts[-1]
    if isinstance(target, list):
        target[int(last)] = value
    else:
        target[last] = value


async def create_thread(document: dict[str, Any]) -> dict[str, Any]:
    thread = dict(document)
    thread["id"] = uuid4().hex
    updated_at = thread.get("updated_at")
    updated_at_text = updated_at.isoformat() if isinstance(updated_at, datetime) else str(updated_at or "")
    async with _lock:
        with _connect() as connection:
            connection.execute(
                "INSERT INTO threads (id, updated_at, document) VALUES (?, ?, ?)",
                (thread["id"], updated_at_text, _dump(thread)),
            )
    return _load(_dump(thread))


async def list_threads() -> list[dict[str, Any]]:
    async with _lock:
        with _connect() as connection:
            rows = connection.execute("SELECT document FROM threads ORDER BY updated_at DESC").fetchall()
    return [_load(row[0]) for row in rows]


async def get_thread(thread_id: str) -> dict[str, Any] | None:
    async with _lock:
        with _connect() as connection:
            row = connection.execute("SELECT document FROM threads WHERE id = ?", (thread_id,)).fetchone()
    return _load(row[0]) if row else None


async def update_thread_fields(thread_id: str, fields: dict[str, Any]) -> dict[str, Any] | None:
    async with _lock:
        with _connect() as connection:
            row = connection.execute("SELECT document FROM threads WHERE id = ?", (thread_id,)).fetchone()
            if not row:
                return None
            thread = _load(row[0])
            for path, value in fields.items():
                _set_path(thread, path, value)
            updated_at = thread.get("updated_at")
            updated_at_text = updated_at.isoformat() if isinstance(updated_at, datetime) else str(updated_at or "")
            connection.execute(
                "UPDATE threads SET updated_at = ?, document = ? WHERE id = ?",
                (updated_at_text, _dump(thread), thread_id),
            )
        return _load(_dump(thread))


async def delete_thread(thread_id: str) -> bool:
    async with _lock:
        with _connect() as connection:
            cursor = connection.execute("DELETE FROM threads WHERE id = ?", (thread_id,))
            return cursor.rowcount > 0
