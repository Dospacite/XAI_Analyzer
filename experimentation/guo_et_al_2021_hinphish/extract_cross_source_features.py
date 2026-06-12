#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import random
import sys
import zipfile
from pathlib import Path
from typing import Any, Iterable

try:
    from tqdm import tqdm
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    def tqdm(iterable=None, **_: object):
        return iterable


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from features import common as c
from features.extractor import build_feature_record


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCES = {
    "phishing": {
        "archive": SCRIPT_DIR / "phish_sample_30k.zip",
        "label": "phishing",
        "collection": "phish_sample_30k",
    },
    "benign": {
        "archive": SCRIPT_DIR / "benign_sample_30k.zip",
        "label": "benign",
        "collection": "benign_sample_30k",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract the shared explainable phishing feature set from the HinPhish 30k ZIP datasets."
    )
    parser.add_argument(
        "--source",
        choices=["all", "phishing", "benign"],
        default="all",
        help="Which 30k ZIP dataset to process.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=SCRIPT_DIR / "cross_source_features_30k.jsonl",
        help="Output JSONL path.",
    )
    parser.add_argument("--phishing-zip", type=Path, default=DEFAULT_SOURCES["phishing"]["archive"])
    parser.add_argument("--benign-zip", type=Path, default=DEFAULT_SOURCES["benign"]["archive"])
    parser.add_argument("--limit", type=int, default=0, help="Maximum emitted pages per selected ZIP. 0 means no cap.")
    parser.add_argument("--seed", type=int, default=3407, help="Seed used when --limit requires random sampling.")
    parser.add_argument(
        "--max-html-chars",
        type=int,
        default=5_000_000,
        help="Skip pages whose html.txt exceeds this many decoded characters. Use 0 for no cap.",
    )
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def selected_sources(args: argparse.Namespace) -> list[tuple[str, Path, str, str]]:
    configured = {
        "phishing": (args.phishing_zip, DEFAULT_SOURCES["phishing"]["label"], DEFAULT_SOURCES["phishing"]["collection"]),
        "benign": (args.benign_zip, DEFAULT_SOURCES["benign"]["label"], DEFAULT_SOURCES["benign"]["collection"]),
    }
    names = ["phishing", "benign"] if args.source == "all" else [args.source]
    return [(name, *configured[name]) for name in names]


def decode_text(raw: bytes) -> str:
    for encoding in ("utf-8", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def page_dirs(zip_file: zipfile.ZipFile) -> list[str]:
    dirs: set[str] = set()
    for name in zip_file.namelist():
        if name.endswith("/html.txt") and "/" in name:
            dirs.add(name.rsplit("/", 1)[0])
    return sorted(dirs)


def sample_dirs(dirs: list[str], limit: int, seed: int) -> list[str]:
    if limit <= 0 or limit >= len(dirs):
        return dirs
    rng = random.Random(seed)
    return rng.sample(dirs, limit)


def parse_info(info_text: str, page_dir: str, label: str) -> dict[str, Any]:
    text = info_text.strip()
    if not text:
        return {}
    if text.startswith("{") and text.endswith("}"):
        try:
            parsed = ast.literal_eval(text)
            return parsed if isinstance(parsed, dict) else {}
        except (SyntaxError, ValueError):
            return {}
    return {"url": text}


def url_from_info(info: dict[str, Any], page_dir: str, label: str) -> str:
    url = c.collapse_ws(info.get("url"), 2000)
    if url:
        return url
    if label == "benign":
        host = page_dir.strip("/")
        if host.startswith(("http://", "https://")):
            return host
        return f"https://{host}"
    return ""


def read_member(zip_file: zipfile.ZipFile, name: str) -> str:
    try:
        return decode_text(zip_file.read(name))
    except KeyError:
        return ""


def iter_zip_records(
    zip_path: Path,
    label: str,
    collection_name: str,
    args: argparse.Namespace,
) -> Iterable[dict[str, Any]]:
    with zipfile.ZipFile(zip_path) as zip_file:
        dirs = sample_dirs(page_dirs(zip_file), args.limit, args.seed)
        progress = tqdm(
            dirs,
            desc=zip_path.name,
            unit="page",
            total=len(dirs),
            disable=args.no_progress,
            file=sys.stderr,
        )
        skipped_empty = 0
        skipped_large = 0
        try:
            for page_dir in progress:
                html = read_member(zip_file, f"{page_dir}/html.txt")
                if not html.strip():
                    skipped_empty += 1
                    continue
                if args.max_html_chars > 0 and len(html) > args.max_html_chars:
                    skipped_large += 1
                    continue
                info = parse_info(read_member(zip_file, f"{page_dir}/info.txt"), page_dir, label)
                url = url_from_info(info, page_dir, label)
                doc = {
                    "_id": f"{collection_name}:{page_dir}",
                    "url": url,
                    "title": None,
                    "html": html,
                    "metadata": {
                        "url": url,
                        "final_url": url,
                        "status_code": 200,
                        "redirect_history": [],
                    },
                }
                if hasattr(progress, "set_postfix"):
                    progress.set_postfix(skipped_empty=skipped_empty, skipped_large=skipped_large, refresh=False)
                yield doc
        finally:
            close = getattr(progress, "close", None)
            if callable(close):
                close()


def main() -> int:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    with args.output.open("w", encoding="utf-8") as handle:
        for source_name, zip_path, label, collection_name in selected_sources(args):
            if not zip_path.exists():
                print(f"Missing ZIP for {source_name}: {zip_path}", file=sys.stderr)
                return 2
            source_key = f"guo_et_al_2021_hinphish.{collection_name}"
            counts[source_key] = 0
            for doc in iter_zip_records(zip_path, label, collection_name, args):
                record = build_feature_record(
                    doc,
                    "guo_et_al_2021_hinphish",
                    collection_name,
                    label,
                    {"rule": f"{zip_path.name}:{label}", "latest_scan": None},
                )
                c.write_json_line(handle, record)
                counts[source_key] += 1

    total = sum(counts.values())
    print(f"Wrote {total} records to {args.output}", file=sys.stderr)
    for source, count in counts.items():
        print(f"- {source}: {count}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
