#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from pymongo import MongoClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from features import common as c
from features.extractor import build_feature_record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract cross-source phishing features for one MongoDB website document."
    )
    parser.add_argument("--db", required=True, help="MongoDB database name.")
    parser.add_argument("--collection", required=True, help="MongoDB collection name.")
    lookup = parser.add_mutually_exclusive_group(required=True)
    lookup.add_argument("--id", dest="mongo_id", help="MongoDB _id value.")
    lookup.add_argument("--url", dest="website_url", help="Requested or final website URL.")
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--mongo-uri", default=None, help="Mongo URI. Defaults to MONGO_URI from env file.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_dotenv(args.env_file)
    mongo_uri = args.mongo_uri or os.environ.get("MONGO_URI")
    if not mongo_uri:
        print(f"MONGO_URI was not found in {args.env_file}", file=sys.stderr)
        return 2

    query = c.build_lookup_query(args.mongo_id, args.website_url)
    with MongoClient(mongo_uri, serverSelectionTimeoutMS=10000) as client:
        doc = client[args.db][args.collection].find_one(query)
        if not doc:
            print("No matching document found.", file=sys.stderr)
            return 1
        label, label_info = c.infer_dataset_label(args.db, args.collection, doc)
        record = build_feature_record(doc, args.db, args.collection, label, label_info)

    if args.pretty:
        print(json.dumps(record, ensure_ascii=False, indent=2, default=c.json_default))
    else:
        print(json.dumps(record, ensure_ascii=False, default=c.json_default, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
