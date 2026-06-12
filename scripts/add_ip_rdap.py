#!/usr/bin/env python3
"""Backfill IP RDAP data on MongoDB website documents.

Targets:
- tranco.websites
- phishing_db.website_content
- urlscan.live
"""

from __future__ import annotations

import argparse
import ipaddress
import json
import os
import socket
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[1]

SOURCE_COLLECTIONS = {
    "tranco": ("tranco", "websites"),
    "phishing": ("phishing_db", "website_content"),
    "urlscan": ("urlscan", "live"),
}

PHISHING_IP_FIELDS = ("ip", "ips", "ip_address", "ip_addresses")
RDAP_URL_TEMPLATE = "https://rdap.org/ip/{ip}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add ip_rdap RDAP enrichment to website MongoDB documents."
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=ROOT / ".env",
        help="Path to .env containing MONGO_URI.",
    )
    parser.add_argument(
        "--mongo-uri",
        default=None,
        help="MongoDB URI. Defaults to MONGO_URI from --env-file/environment.",
    )
    parser.add_argument(
        "--source",
        choices=["all", *SOURCE_COLLECTIONS.keys()],
        default="all",
        help="Source collection to process.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="MongoDB cursor batch size.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum documents to update per collection. 0 means no limit.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print intended updates without writing to MongoDB.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Refresh documents even when ip_rdap already exists.",
    )
    parser.add_argument(
        "--rdap-timeout",
        type=float,
        default=20.0,
        help="Timeout in seconds for RDAP HTTP requests.",
    )
    parser.add_argument(
        "--dns-timeout",
        type=float,
        default=5.0,
        help="Timeout in seconds for tranco DNS resolution.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.2,
        help="Delay between uncached RDAP HTTP requests.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bars.",
    )
    return parser.parse_args()


def load_env_file(path: Path) -> None:
    try:
        from dotenv import load_dotenv
    except ModuleNotFoundError:
        load_dotenv = None

    if load_dotenv is not None:
        load_dotenv(path)
        return

    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if key:
            os.environ.setdefault(key, value)


def load_mongo_client(mongo_uri: str):
    try:
        from pymongo import MongoClient
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "pymongo is required to update MongoDB. Install it in the runtime environment."
        ) from exc

    return MongoClient(mongo_uri, serverSelectionTimeoutMS=10000)


def progress_wrapper(iterable: Iterable[Any], **kwargs: Any) -> Iterable[Any]:
    try:
        from tqdm import tqdm
    except ModuleNotFoundError:
        return iterable
    return tqdm(iterable, **kwargs)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def first_non_empty(*values: Any) -> Any:
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        if isinstance(value, (dict, list, tuple, set)) and not value:
            continue
        return value
    return None


def safe_get(container: Any, *path: str) -> Any:
    value = container
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def hostname_from_url(url_or_host: Any) -> str:
    text = str(url_or_host or "").strip()
    if not text:
        return ""

    try:
        parsed = urlparse(text)
    except ValueError:
        return ""

    if parsed.hostname:
        return parsed.hostname.lower().strip(".")

    try:
        parsed = urlparse(f"//{text}")
    except ValueError:
        return ""
    return (parsed.hostname or "").lower().strip(".")


def normalize_ip(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().strip("[]")
    if not text:
        return None
    try:
        return str(ipaddress.ip_address(text))
    except ValueError:
        return None


def iter_ip_candidates(value: Any) -> Iterable[str]:
    if value is None:
        return

    if isinstance(value, dict):
        for key in ("ip", "address", "value"):
            normalized = normalize_ip(value.get(key))
            if normalized:
                yield normalized
        for nested in value.values():
            yield from iter_ip_candidates(nested)
        return

    if isinstance(value, (list, tuple, set)):
        for item in value:
            yield from iter_ip_candidates(item)
        return

    normalized = normalize_ip(value)
    if normalized:
        yield normalized
        return

    if isinstance(value, str):
        for token in value.replace(",", " ").replace(";", " ").split():
            normalized = normalize_ip(token)
            if normalized:
                yield normalized


def first_ip_from_values(*values: Any) -> str | None:
    for value in values:
        for ip in iter_ip_candidates(value):
            return ip
    return None


def resolve_first_ip(host_or_ip: str, timeout: float) -> str:
    normalized = normalize_ip(host_or_ip)
    if normalized:
        return normalized

    host = hostname_from_url(host_or_ip) or str(host_or_ip or "").strip()
    if not host:
        raise LookupError("missing hostname")

    old_timeout = socket.getdefaulttimeout()
    socket.setdefaulttimeout(timeout)
    try:
        results = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise LookupError(str(exc)) from exc
    except TimeoutError as exc:
        raise LookupError(f"DNS lookup timed out after {timeout:g}s") from exc
    finally:
        socket.setdefaulttimeout(old_timeout)

    for result in results:
        sockaddr = result[4]
        if not sockaddr:
            continue
        ip = normalize_ip(sockaddr[0])
        if ip:
            return ip

    raise LookupError(f"DNS lookup returned no usable IP for {host}")


def rdap_lookup(ip: str, timeout: float) -> dict[str, Any]:
    try:
        import requests
    except ModuleNotFoundError as exc:
        raise RuntimeError("requests is required for RDAP HTTP lookups.") from exc

    url = RDAP_URL_TEMPLATE.format(ip=ip)
    response = requests.get(url, timeout=timeout, headers={"Accept": "application/rdap+json, application/json"})
    try:
        body = response.json()
    except ValueError as exc:
        raise ValueError(f"RDAP response was not JSON: HTTP {response.status_code}") from exc

    if response.status_code >= 400:
        message = body.get("description") if isinstance(body, dict) else None
        if isinstance(message, list):
            message = " ".join(str(item) for item in message)
        raise RuntimeError(f"RDAP HTTP {response.status_code}: {message or response.text[:200]}")

    if not isinstance(body, dict):
        raise ValueError("RDAP response JSON was not an object")
    return body


def ok_payload(ip: str, ip_source: str, rdap: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": "ok",
        "ip": ip,
        "ip_source": ip_source,
        "rdap": rdap,
        "looked_up_at": utc_now_iso(),
    }


def error_payload(
    ip_source: str,
    error_type: str,
    error: str,
    ip: str | None = None,
) -> dict[str, Any]:
    return {
        "status": "error",
        "ip": ip,
        "ip_source": ip_source,
        "error_type": error_type,
        "error": error[:1000],
        "looked_up_at": utc_now_iso(),
    }


def tranco_ip(doc: dict[str, Any], dns_timeout: float) -> tuple[str | None, str, dict[str, Any] | None]:
    url = first_non_empty(safe_get(doc, "metadata", "final_url"), doc.get("url"))
    host = hostname_from_url(url)
    if not host:
        return None, "dns_first", error_payload("dns_first", "missing_ip", "missing URL hostname")
    try:
        return resolve_first_ip(host, dns_timeout), "dns_first", None
    except LookupError as exc:
        return None, "dns_first", error_payload("dns_first", "dns_error", str(exc))


def phishing_final_url(doc: dict[str, Any]) -> str | None:
    value = safe_get(doc, "metadata", "final_url")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def phishing_ip(doc: dict[str, Any], phishing_urls: Any) -> tuple[str | None, str, dict[str, Any] | None]:
    final_url = phishing_final_url(doc)
    if not final_url:
        return None, "phishing_urls", error_payload(
            "phishing_urls",
            "missing_ip",
            "website_content.metadata.final_url is missing",
        )

    projection = {"_id": 0, "url": 1}
    for field in PHISHING_IP_FIELDS:
        projection[field] = 1

    label_doc = phishing_urls.find_one({"url": final_url}, projection)
    if not label_doc:
        return None, "phishing_urls", error_payload(
            "phishing_urls",
            "missing_ip",
            f"no phishing_urls row matched metadata.final_url={final_url!r}",
        )

    values = [label_doc.get(field) for field in PHISHING_IP_FIELDS]
    ip = first_ip_from_values(*values)
    if not ip:
        return None, "phishing_urls", error_payload(
            "phishing_urls",
            "missing_ip",
            f"matched phishing_urls row has no usable IP in {', '.join(PHISHING_IP_FIELDS)}",
        )
    return ip, "phishing_urls", None


def urlscan_ip(doc: dict[str, Any]) -> tuple[str | None, str, dict[str, Any] | None]:
    ip = first_ip_from_values(
        safe_get(doc, "page", "ip"),
        safe_get(doc, "urlscanresults", "page", "ip"),
    )
    if not ip:
        return None, "urlscan_page", error_payload(
            "urlscan_page",
            "missing_ip",
            "page.ip and urlscanresults.page.ip are missing or invalid",
        )
    return ip, "urlscan_page", None


def extract_ip(
    source: str,
    doc: dict[str, Any],
    client: Any,
    dns_timeout: float,
) -> tuple[str | None, str, dict[str, Any] | None]:
    if source == "tranco":
        return tranco_ip(doc, dns_timeout)
    if source == "phishing":
        return phishing_ip(doc, client["phishing_db"]["phishing_urls"])
    if source == "urlscan":
        return urlscan_ip(doc)
    raise ValueError(f"unknown source: {source}")


def source_projection(source: str) -> dict[str, int]:
    projection = {"_id": 1, "ip_rdap": 1}
    if source == "tranco":
        projection.update({"url": 1, "metadata.final_url": 1})
    elif source == "phishing":
        projection.update({"metadata.final_url": 1})
    elif source == "urlscan":
        projection.update({"page.ip": 1, "urlscanresults.page.ip": 1})
    return projection


def selected_sources(source: str) -> list[tuple[str, str, str]]:
    names = SOURCE_COLLECTIONS.keys() if source == "all" else [source]
    return [(name, *SOURCE_COLLECTIONS[name]) for name in names]


def update_limit_reached(stats: Counter[str], limit: int) -> bool:
    return limit > 0 and stats["updated"] + stats["dry_run"] >= limit


def process_collection(
    client: Any,
    source: str,
    db_name: str,
    collection_name: str,
    args: argparse.Namespace,
    rdap_cache: dict[str, dict[str, Any]],
) -> Counter[str]:
    collection = client[db_name][collection_name]
    query: dict[str, Any] = {} if args.force else {"ip_rdap": {"$exists": False}}
    cursor = collection.find(
        query,
        source_projection(source),
        batch_size=args.batch_size,
        no_cursor_timeout=False,
    ).sort("_id", 1)

    stats: Counter[str] = Counter()
    progress = progress_wrapper(
        cursor,
        desc=f"{db_name}.{collection_name}",
        unit="doc",
        disable=args.no_progress,
        file=sys.stderr,
    )

    for doc in progress:
        if update_limit_reached(stats, args.limit):
            break

        stats["scanned"] += 1
        if not args.force and "ip_rdap" in doc:
            stats["skipped_existing"] += 1
            continue

        ip, ip_source, extraction_error = extract_ip(source, doc, client, args.dns_timeout)
        if extraction_error is not None:
            payload = extraction_error
            error_type = payload.get("error_type")
            if error_type == "dns_error":
                stats["dns_errors"] += 1
            elif error_type == "invalid_ip":
                stats["invalid_ip"] += 1
            else:
                stats["missing_ip"] += 1
        elif ip is None:
            payload = error_payload(ip_source, "missing_ip", "IP extraction returned no IP")
            stats["missing_ip"] += 1
        else:
            cached = rdap_cache.get(ip)
            if cached is None:
                if args.sleep_seconds > 0 and stats["rdap_requests"] > 0:
                    time.sleep(args.sleep_seconds)
                try:
                    rdap = rdap_lookup(ip, args.rdap_timeout)
                except RuntimeError as exc:
                    payload = error_payload(ip_source, "rdap_http_error", str(exc), ip=ip)
                    stats["rdap_errors"] += 1
                except ValueError as exc:
                    payload = error_payload(ip_source, "rdap_parse_error", str(exc), ip=ip)
                    stats["rdap_errors"] += 1
                except Exception as exc:  # pragma: no cover - network libraries vary
                    payload = error_payload(ip_source, "rdap_http_error", str(exc), ip=ip)
                    stats["rdap_errors"] += 1
                else:
                    payload = ok_payload(ip, ip_source, rdap)
                    rdap_cache[ip] = payload
                    stats["rdap_ok"] += 1
                stats["rdap_requests"] += 1
            else:
                payload = dict(cached)
                payload["ip_source"] = ip_source
                payload["looked_up_at"] = utc_now_iso()
                stats["rdap_cache_hits"] += 1

        if args.dry_run:
            stats["dry_run"] += 1
            print(
                json.dumps(
                    {
                        "source": f"{db_name}.{collection_name}",
                        "_id": str(doc.get("_id")),
                        "ip_rdap": payload,
                    },
                    ensure_ascii=False,
                    default=str,
                )
            )
        else:
            collection.update_one({"_id": doc["_id"]}, {"$set": {"ip_rdap": payload}})
            stats["updated"] += 1

    return stats


def print_summary(source_name: str, db_name: str, collection_name: str, stats: Counter[str]) -> None:
    summary_keys = (
        "scanned",
        "updated",
        "dry_run",
        "skipped_existing",
        "missing_ip",
        "dns_errors",
        "invalid_ip",
        "rdap_ok",
        "rdap_errors",
        "rdap_requests",
        "rdap_cache_hits",
    )
    summary = " ".join(f"{key}={stats[key]}" for key in summary_keys if stats[key])
    if not summary:
        summary = "scanned=0"
    print(f"{source_name} ({db_name}.{collection_name}): {summary}", file=sys.stderr)


def main() -> int:
    args = parse_args()
    load_env_file(args.env_file)

    mongo_uri = args.mongo_uri or os.environ.get("MONGO_URI")
    if not mongo_uri:
        print(f"MONGO_URI was not found in {args.env_file}", file=sys.stderr)
        return 2

    try:
        client = load_mongo_client(mongo_uri)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    rdap_cache: dict[str, dict[str, Any]] = {}
    total: Counter[str] = Counter()

    try:
        client.admin.command("ping")
        for source_name, db_name, collection_name in selected_sources(args.source):
            stats = process_collection(
                client,
                source_name,
                db_name,
                collection_name,
                args,
                rdap_cache,
            )
            total.update(stats)
            print_summary(source_name, db_name, collection_name, stats)
    finally:
        client.close()

    print_summary("total", "*", "*", total)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
