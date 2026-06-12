#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from dotenv import load_dotenv
from pymongo import MongoClient

try:
    from tqdm import tqdm
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    class tqdm:  # type: ignore[no-redef]
        def __init__(self, iterable=None, **_: object):
            self.iterable = iterable

        def __iter__(self):
            return iter(self.iterable or [])

        def update(self, *_: object, **__: object) -> None:
            return None

        def set_postfix(self, *_: object, **__: object) -> None:
            return None

        def close(self) -> None:
            return None

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from features import common as c
from features.extractor import build_feature_record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract cross-source phishing features from MongoDB documents into JSONL."
    )
    parser.add_argument(
        "--source",
        choices=["all", "phishing", "benign", "urlscan_live"],
        default="all",
        help="Named source set. Ignored when --db and --collection are provided.",
    )
    parser.add_argument("--db", default=None, help="MongoDB database name for a custom collection.")
    parser.add_argument("--collection", default=None, help="MongoDB collection name for a custom collection.")
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL path.")
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--mongo-uri", default=None, help="Mongo URI. Defaults to MONGO_URI from env file.")
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--limit", type=int, default=0, help="Maximum documents per collection. 0 means no cap.")
    parser.add_argument(
        "--max-scanned-per-collection",
        type=int,
        default=0,
        help="Stop after scanning this many source documents per collection. 0 means no cap.",
    )
    parser.add_argument(
        "--max-html-chars",
        type=int,
        default=5_000_000,
        help="Skip documents whose decoded HTML exceeds this many characters. Use 0 for no cap.",
    )
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def selected_collections(args: argparse.Namespace) -> list[tuple[str, str]]:
    if args.db or args.collection:
        if not args.db or not args.collection:
            raise ValueError("--db and --collection must be provided together.")
        return [(args.db, args.collection)]
    if args.source == "all":
        return list(c.DEFAULT_COLLECTIONS.values())
    return [c.DEFAULT_COLLECTIONS[args.source]]


def random_sample_size(args: argparse.Namespace, collection_count: int) -> int:
    if args.max_scanned_per_collection > 0:
        return min(args.max_scanned_per_collection, collection_count)
    return min(collection_count, args.limit * 3)


def iter_random_candidates(collection: Any, base_query: dict[str, Any], args: argparse.Namespace, sample_size: int) -> Iterable[dict]:
    yield from collection.aggregate(
        [
            {"$sample": {"size": sample_size}},
            {"$match": base_query},
            {"$project": c.mongo_feature_projection()},
        ],
        batchSize=args.batch_size,
        allowDiskUse=True,
    )


def iter_sequential_candidates(collection: Any, base_query: dict[str, Any], args: argparse.Namespace) -> Iterable[dict]:
    last_id = None
    while True:
        query = dict(base_query)
        if last_id is not None:
            query = {"$and": [base_query, {"_id": {"$gt": last_id}}]}
        cursor = (
            collection.find(
                query,
                c.mongo_feature_projection(),
                batch_size=args.batch_size,
                no_cursor_timeout=False,
            )
            .sort("_id", 1)
            .limit(args.batch_size)
        )
        batch_count = 0
        for doc in cursor:
            batch_count += 1
            last_id = doc.get("_id")
            yield doc
        if batch_count == 0:
            return


def iter_documents(client: MongoClient, db_name: str, collection_name: str, args: argparse.Namespace) -> Iterable[tuple[dict, str, dict[str, Any]]]:
    collection = client[db_name][collection_name]
    base_query = c.mongo_html_query()
    collection_count: int | None = None
    random_sample = False
    sample_size = 0
    progress_total: int | None = None
    if args.limit > 0:
        collection_count = collection.estimated_document_count()
        random_sample = args.limit < collection_count
        sample_size = random_sample_size(args, collection_count) if random_sample else 0
        progress_total = min(args.limit, collection_count)

    scanned = 0
    yielded = 0
    skipped_empty = 0
    skipped_large = 0
    skipped_status = 0
    skipped_generic_error = 0
    skipped_unlabeled = 0
    candidates = (
        iter_random_candidates(collection, base_query, args, sample_size)
        if random_sample
        else iter_sequential_candidates(collection, base_query, args)
    )
    progress = tqdm(
        desc=f"{db_name}.{collection_name}",
        total=progress_total,
        unit="doc",
        disable=args.no_progress,
        file=sys.stderr,
    )
    if random_sample and hasattr(progress, "set_postfix"):
        progress.set_postfix(scanned=0, sampled=sample_size, refresh=False)
    try:
        for doc in candidates:
            scanned += 1
            if args.max_scanned_per_collection > 0 and scanned > args.max_scanned_per_collection:
                return
            normalized = c.normalize_document(doc)
            html = normalized.get("html") or ""
            status = c.normalized_status_code(normalized)
            if status != 200:
                skipped_status += 1
                continue
            if not html.strip():
                skipped_empty += 1
                continue
            title = c.title_from_doc_or_html(normalized, c.html_selector(html))
            if c.collapse_ws(title, 200).lower() in c.GENERIC_ERROR_TITLES:
                skipped_generic_error += 1
                continue
            if args.max_html_chars > 0 and len(html) > args.max_html_chars:
                skipped_large += 1
                continue
            label, label_info = c.infer_dataset_label(db_name, collection_name, doc)
            if label not in {"phishing", "benign"}:
                skipped_unlabeled += 1
                continue
            yielded += 1
            if (yielded == 1 or scanned % 100 == 0 or (args.limit > 0 and yielded >= args.limit)) and hasattr(progress, "set_postfix"):
                postfix = {
                    "scanned": scanned,
                    "skipped_empty": skipped_empty,
                    "skipped_large": skipped_large,
                    "skipped_status": skipped_status,
                    "skipped_generic_error": skipped_generic_error,
                    "skipped_unlabeled": skipped_unlabeled,
                }
                if random_sample:
                    postfix["sampled"] = sample_size
                progress.set_postfix(
                    **postfix,
                    refresh=False,
                )
            progress.update(1)
            yield doc, label, label_info
            if args.limit > 0 and yielded >= args.limit:
                return
    finally:
        progress.close()


def iter_documents_cursor(client: MongoClient, db_name: str, collection_name: str, args: argparse.Namespace) -> Iterable[dict]:
    cursor = client[db_name][collection_name].find(
        c.mongo_html_query(),
        c.mongo_feature_projection(),
        batch_size=args.batch_size,
    )
    progress = tqdm(
        cursor,
        desc=f"{db_name}.{collection_name}",
        unit="doc",
        disable=args.no_progress,
        file=sys.stderr,
    )
    scanned = 0
    skipped_empty = 0
    skipped_large = 0
    skipped_status = 0
    skipped_generic_error = 0
    for doc in progress:
        scanned += 1
        if args.max_scanned_per_collection > 0 and scanned > args.max_scanned_per_collection:
            break
        normalized = c.normalize_document(doc)
        html = normalized.get("html") or ""
        status = c.normalized_status_code(normalized)
        if status != 200:
            skipped_status += 1
            continue
        if not html.strip():
            skipped_empty += 1
            continue
        title = c.title_from_doc_or_html(normalized, c.html_selector(html))
        if c.collapse_ws(title, 200).lower() in c.GENERIC_ERROR_TITLES:
            skipped_generic_error += 1
            continue
        if args.max_html_chars > 0 and len(html) > args.max_html_chars:
            skipped_large += 1
            continue
        if scanned % 100 == 0 and hasattr(progress, "set_postfix"):
            progress.set_postfix(
                scanned=scanned,
                skipped_empty=skipped_empty,
                skipped_large=skipped_large,
                skipped_status=skipped_status,
                skipped_generic_error=skipped_generic_error,
                refresh=False,
            )
        yield doc


def main() -> int:
    args = parse_args()
    load_dotenv(args.env_file)
    mongo_uri = args.mongo_uri or os.environ.get("MONGO_URI")
    if not mongo_uri:
        print(f"MONGO_URI was not found in {args.env_file}", file=sys.stderr)
        return 2

    args.output.parent.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    with MongoClient(mongo_uri, serverSelectionTimeoutMS=10000) as client, args.output.open("w", encoding="utf-8") as handle:
        for db_name, collection_name in selected_collections(args):
            source_key = f"{db_name}.{collection_name}"
            for doc, label, label_info in iter_documents(client, db_name, collection_name, args):
                record = build_feature_record(doc, db_name, collection_name, label, label_info)
                c.write_json_line(handle, record)
                counts[source_key] += 1
                if args.limit > 0 and counts[source_key] >= args.limit:
                    break

    total = sum(counts.values())
    print(f"Wrote {total} records to {args.output}", file=sys.stderr)
    for source, count in counts.items():
        print(f"- {source}: {count}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
