from __future__ import annotations

from functools import lru_cache
from importlib import resources
from pathlib import Path

import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict

from nim_model_router.types import ClassifierConfig, Registry, TaskConfig

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEV_REGISTRY_PATH = PACKAGE_ROOT / "config" / "models.yaml"


def default_registry_path() -> Path:
    bundled = Path(__file__).resolve().parent / "models.yaml"
    if bundled.exists():
        return bundled
    if DEV_REGISTRY_PATH.exists():
        return DEV_REGISTRY_PATH
    return Path(
        str(resources.files("nim_model_router").joinpath("models.yaml"))
    )


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
    nvcf_poll_seconds: str = "300"


def load_registry(path: Path | None = None) -> Registry:
    config_path = path or DEFAULT_REGISTRY_PATH
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    tasks = {name: TaskConfig(**cfg) for name, cfg in raw.get("tasks", {}).items()}
    aliases = {k: v for k, v in raw.get("aliases", {}).items()}
    classifier = ClassifierConfig(**raw.get("classifier", {}))
    return Registry(tasks=tasks, aliases=aliases, classifier=classifier)


@lru_cache
def get_settings() -> Settings:
    return Settings()