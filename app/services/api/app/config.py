from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    sqlite_path: str = "traceguard.sqlite3"
    scraper_url: str = "http://scraper:8081"
    scraper_token: str = "local-development-token"

    hf_token: str = ""
    hf_qwen_repo_id: str = "Dospacite/xai-phishing-qwen3-4b-merged"
    hf_llama_repo_id: str = "Dospacite/xai-phishing-llama-3.1-8b-merged"
    hf_deepseek_repo_id: str = "Dospacite/xai-phishing-deepseek-r1-qwen-7b-merged"
    hf_gemma_repo_id: str = "Dospacite/gemma4-e4b-unsloth-phishing-merged"
    hf_qwen_endpoint_url: str = ""
    hf_llama_endpoint_url: str = ""
    hf_deepseek_endpoint_url: str = ""
    hf_gemma_endpoint_url: str = ""
    hf_endpoint_namespace: str = "Dospacite"
    hf_manage_endpoint_lifecycle: bool = True
    hf_endpoint_start_timeout_seconds: float = 900
    hf_endpoint_poll_seconds: float = 10
    hf_qwen_endpoint_name: str = "traceguard-qwen3-4b"
    hf_llama_endpoint_name: str = "traceguard-llama-3-1-8b"
    hf_deepseek_endpoint_name: str = "traceguard-deepseek-r1-qwen-7b"
    hf_gemma_endpoint_name: str = "traceguard-gemma4-e4b"

    qwen_api_key: str = ""
    qwen_base_url: str = ""
    qwen_model: str = "qwen3.5-flash"

    public_app_url: str = "http://localhost:3000"
    request_timeout_seconds: float = 180

    model_config = SettingsConfigDict(env_file=None, case_sensitive=False, extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()
