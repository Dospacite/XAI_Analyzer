from datetime import datetime, timezone

import pytest

from app.store import create_thread, delete_thread, get_thread, init_store, list_threads, update_thread_fields


@pytest.mark.asyncio
async def test_sqlite_store_persists_threads_across_reinitialization(tmp_path):
    db_path = tmp_path / "traceguard.sqlite3"
    await init_store(str(db_path))

    created = await create_thread(
        {
            "title": "Example",
            "url": "https://example.com",
            "status": "processing",
            "overall_verdict": "unknown",
            "progress": "Queued",
            "analyses": [{"status": "pending", "error": None}],
            "messages": [],
            "qwen_history": [],
            "updated_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
        }
    )

    await update_thread_fields(
        created["id"],
        {
            "status": "ready",
            "analyses.0.status": "complete",
            "updated_at": datetime(2026, 1, 2, tzinfo=timezone.utc),
        },
    )

    await init_store(str(db_path))
    loaded = await get_thread(created["id"])

    assert loaded is not None
    assert loaded["status"] == "ready"
    assert loaded["analyses"][0]["status"] == "complete"
    assert loaded["updated_at"] == datetime(2026, 1, 2, tzinfo=timezone.utc)
    assert [item["id"] for item in await list_threads()] == [created["id"]]

    assert await delete_thread(created["id"]) is True
    assert await get_thread(created["id"]) is None
