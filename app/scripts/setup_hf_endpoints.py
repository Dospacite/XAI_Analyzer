#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path


DEFAULT_MODELS = {
    "QWEN": {
        "repo": "Dospacite/xai-phishing-qwen3-4b-merged",
        "name": "traceguard-qwen3-4b",
        "instance_type": "nvidia-l4",
        "instance_size": "x1",
    },
    "LLAMA": {
        "repo": "Dospacite/xai-phishing-llama-3.1-8b-merged",
        "name": "traceguard-llama-3-1-8b",
        "instance_type": "nvidia-l40s",
        "instance_size": "x1",
    },
    "DEEPSEEK": {
        "repo": "Dospacite/xai-phishing-deepseek-r1-qwen-7b-merged",
        "name": "traceguard-deepseek-r1-qwen-7b",
        "instance_type": "nvidia-l4",
        "instance_size": "x1",
    },
    "GEMMA": {
        "repo": "Dospacite/gemma4-e4b-unsloth-phishing-merged",
        "name": "traceguard-gemma4-e4b",
        "instance_type": "nvidia-l40s",
        "instance_size": "x1",
    },
}


@dataclass(frozen=True)
class EndpointConfig:
    key: str
    name: str
    repository: str
    framework: str
    task: str
    accelerator: str
    vendor: str
    region: str
    instance_type: str
    instance_size: str
    endpoint_type: str
    namespace: str
    min_replica: int
    max_replica: int
    scale_to_zero_timeout: int | None


def parse_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        values[key] = value
    return values


def env_value(values: dict[str, str], key: str, default: str = "") -> str:
    return os.environ.get(key) or values.get(key) or default


def env_int(values: dict[str, str], key: str, default: int) -> int:
    raw_value = env_value(values, key, str(default))
    try:
        return int(raw_value)
    except ValueError as exc:
        raise SystemExit(f"{key} must be an integer, got {raw_value!r}") from exc


def env_optional_int(values: dict[str, str], key: str) -> int | None:
    raw_value = env_value(values, key)
    if not raw_value:
        return None
    try:
        return int(raw_value)
    except ValueError as exc:
        raise SystemExit(f"{key} must be an integer, got {raw_value!r}") from exc


def model_value(values: dict[str, str], model_key: str, suffix: str, default: str = "") -> str:
    return env_value(
        values,
        f"HF_{model_key}_ENDPOINT_{suffix}",
        env_value(values, f"HF_ENDPOINT_{suffix}", default),
    )


def build_endpoint_configs(values: dict[str, str]) -> list[EndpointConfig]:
    namespace = env_value(values, "HF_ENDPOINT_NAMESPACE")
    if not namespace:
        raise SystemExit("HF_ENDPOINT_NAMESPACE is required in .env")

    configs: list[EndpointConfig] = []
    for key, defaults in DEFAULT_MODELS.items():
        configs.append(
            EndpointConfig(
                key=key,
                name=env_value(values, f"HF_{key}_ENDPOINT_NAME", defaults["name"]),
                repository=env_value(values, f"HF_{key}_REPO_ID", defaults["repo"]),
                framework=model_value(values, key, "FRAMEWORK", "pytorch"),
                task=model_value(values, key, "TASK", "text-generation"),
                accelerator=model_value(values, key, "ACCELERATOR", "gpu"),
                vendor=model_value(values, key, "VENDOR", "aws"),
                region=model_value(values, key, "REGION", "us-east-1"),
                instance_type=model_value(values, key, "INSTANCE_TYPE", defaults["instance_type"]),
                instance_size=model_value(values, key, "INSTANCE_SIZE", defaults["instance_size"]),
                endpoint_type=model_value(values, key, "TYPE", "protected"),
                namespace=namespace,
                min_replica=env_int(values, f"HF_{key}_ENDPOINT_MIN_REPLICA", env_int(values, "HF_ENDPOINT_MIN_REPLICA", 1)),
                max_replica=env_int(values, f"HF_{key}_ENDPOINT_MAX_REPLICA", env_int(values, "HF_ENDPOINT_MAX_REPLICA", 1)),
                scale_to_zero_timeout=env_optional_int(values, f"HF_{key}_ENDPOINT_SCALE_TO_ZERO_TIMEOUT")
                or env_optional_int(values, "HF_ENDPOINT_SCALE_TO_ZERO_TIMEOUT"),
            )
        )
    return configs


def endpoint_url(endpoint: object) -> str:
    direct_url = getattr(endpoint, "url", None)
    if direct_url:
        return str(direct_url)
    raw = getattr(endpoint, "raw", None)
    if isinstance(raw, dict):
        for key in ("url", "endpointUrl"):
            value = raw.get(key)
            if isinstance(value, str) and value:
                return value
    return ""


def update_env_file(path: Path, updates: dict[str, str]) -> None:
    existing = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    remaining = dict(updates)
    output: list[str] = []

    for line in existing:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in line:
            key = line.split("=", 1)[0].strip()
            if key in remaining:
                output.append(f"{key}={remaining.pop(key)}")
                continue
        output.append(line)

    if remaining:
        if output and output[-1].strip():
            output.append("")
        output.append("# Hugging Face endpoint URLs generated by scripts/setup_hf_endpoints.py")
        for key in sorted(remaining):
            output.append(f"{key}={remaining[key]}")

    path.write_text("\n".join(output) + "\n", encoding="utf-8")


def require_huggingface_hub() -> tuple[object, object, type[Exception]]:
    try:
        from huggingface_hub import create_inference_endpoint, get_inference_endpoint
        from huggingface_hub.utils import HfHubHTTPError
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: huggingface_hub. Install it with:\n"
            "  python3 -m pip install 'huggingface_hub>=0.19.0'"
        ) from exc
    return create_inference_endpoint, get_inference_endpoint, HfHubHTTPError


def create_or_get_endpoint(
    config: EndpointConfig,
    *,
    token: str,
    wait: bool,
    pause_after_create: bool,
) -> tuple[object, bool]:
    create_inference_endpoint, get_inference_endpoint, hf_error = require_huggingface_hub()

    try:
        endpoint = get_inference_endpoint(config.name, namespace=config.namespace, token=token)
        return endpoint, False
    except hf_error as exc:
        status_code = getattr(getattr(exc, "response", None), "status_code", None)
        if status_code != 404:
            raise

    endpoint = create_inference_endpoint(
        config.name,
        repository=config.repository,
        framework=config.framework,
        task=config.task,
        accelerator=config.accelerator,
        vendor=config.vendor,
        region=config.region,
        type=config.endpoint_type,
        instance_size=config.instance_size,
        instance_type=config.instance_type,
        min_replica=config.min_replica,
        max_replica=config.max_replica,
        scale_to_zero_timeout=config.scale_to_zero_timeout,
        namespace=config.namespace,
        token=token,
    )
    if wait:
        endpoint = endpoint.wait()
    if pause_after_create:
        if not wait:
            print(f"Skipping pause for {config.name}; use --wait with --pause-after-create.")
        else:
            endpoint.pause()
            endpoint = get_inference_endpoint(config.name, namespace=config.namespace, token=token)
    return endpoint, True


def print_plan(configs: list[EndpointConfig]) -> None:
    print("Endpoint creation plan:")
    for config in configs:
        print(
            f"- {config.key}: {config.name} -> {config.repository} "
            f"({config.vendor}/{config.region}, {config.instance_type} {config.instance_size})"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create the Hugging Face Inference Endpoints required by Traceguard.",
    )
    parser.add_argument(
        "--env-file",
        default=str(Path(__file__).resolve().parents[1] / ".env"),
        help="Path to the app .env file. Defaults to app/.env.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the endpoint plan without creating anything.",
    )
    parser.add_argument(
        "--wait",
        action="store_true",
        help="Wait for newly created endpoints to become ready. This can take several minutes.",
    )
    parser.add_argument(
        "--pause-after-create",
        action="store_true",
        help="Pause newly created endpoints after they become ready. Requires --wait.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip the billing confirmation prompt.",
    )
    args = parser.parse_args()

    env_path = Path(args.env_file).expanduser().resolve()
    values = parse_env_file(env_path)
    token = env_value(values, "HF_TOKEN")
    if not token or token == "hf_replace_me":
        raise SystemExit(f"HF_TOKEN must be set in {env_path}")

    configs = build_endpoint_configs(values)
    print_plan(configs)

    if args.dry_run:
        return 0

    if args.pause_after_create and not args.wait:
        raise SystemExit("--pause-after-create requires --wait")

    if not args.yes:
        answer = input(
            "\nCreating dedicated Hugging Face Inference Endpoints is billable. Continue? [y/N] "
        ).strip()
        if answer.lower() not in {"y", "yes"}:
            print("Cancelled.")
            return 1

    url_updates: dict[str, str] = {}
    for config in configs:
        print(f"\nSetting up {config.name}...")
        endpoint, created = create_or_get_endpoint(
            config,
            token=token,
            wait=args.wait,
            pause_after_create=args.pause_after_create,
        )
        state = getattr(getattr(endpoint, "status", None), "value", getattr(endpoint, "status", "unknown"))
        url = endpoint_url(endpoint)
        action = "created" if created else "already exists"
        print(f"{config.name}: {action}; status={state}; url={url or 'not assigned yet'}")
        if url:
            url_updates[f"HF_{config.key}_ENDPOINT_URL"] = url

    if url_updates:
        update_env_file(env_path, url_updates)
        print(f"\nUpdated endpoint URLs in {env_path}")
    else:
        print("\nNo endpoint URLs were available yet. Re-run with --wait or copy them from Hugging Face.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
