from __future__ import annotations

from functools import lru_cache
from importlib import resources
from pathlib import Path

import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict

from nim_model_router.types import ClassifierConfig, Registry, RoutePolicies, TaskConfig


def default_registry_path() -> Path:
    bundled = Path(__file__).resolve().parent / "models.yaml"
    if bundled.exists():
        return bundled
    return Path(str(resources.files("nim_model_router").joinpath("models.yaml")))


DEFAULT_REGISTRY_PATH = default_registry_path()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    nvidia_api_key: str = ""
    nim_base_url: str = "https://integrate.api.nvidia.com/v1"
    router_host: str = "127.0.0.1"
    router_port: int = 8080
    router_config: Path = DEFAULT_REGISTRY_PATH
    router_log_path: Path | None = None
    router_api_key: str = ""
    nvcf_poll_seconds: str = "300"
    upstream_max_retries: int = 3
    upstream_retry_backoff_seconds: float = 0.5
    enable_prometheus: bool = True
    max_request_body_bytes: int = 10_485_760
    health_check_upstream: bool = False


def load_registry(path: Path | None = None) -> Registry:
    config_path = path or DEFAULT_REGISTRY_PATH
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    tasks = {name: TaskConfig(**cfg) for name, cfg in raw.get("tasks", {}).items()}
    aliases = {k: v for k, v in raw.get("aliases", {}).items()}
    classifier = ClassifierConfig(**raw.get("classifier", {}))
    policies = RoutePolicies(**raw.get("policies", {}))
    latency_routing = bool(raw.get("latency_routing", True))
    return Registry(
        tasks=tasks,
        aliases=aliases,
        classifier=classifier,
        policies=policies,
        latency_routing=latency_routing,
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
