"""Application settings, loaded from environment variables and an optional ``.env`` file."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(".env", "../.env"),
        extra="ignore",
        case_sensitive=False,
    )

    # Default ArangoDB connection. Every value may be overridden per request with
    # X-Arango-* headers when ``allow_connection_override`` is enabled.
    arango_endpoint: str = "http://localhost:8529"
    arango_database: str = "_system"
    arango_username: str = "root"
    arango_password: SecretStr = SecretStr("")
    arango_verify_ssl: bool = True
    arango_request_timeout: int = 120

    allow_connection_override: bool = True

    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:5173"])
    static_dir: Path | None = None
    max_upload_mb: int = 1024
    upload_dir: Path | None = None

    # Remote sources (POST /load/url). Empty allow-list = any host.
    allow_url_sources: bool = True
    url_allowed_hosts: list[str] = Field(default_factory=list)

    # Embedding defaults
    embed_concurrency: int = 4
    default_batch_size: int = 64
    presets_file: Path | None = None

    # Collection embedding profiles
    profiles_collection: str = "embedding_loader_profiles"
    start_watchers_on_startup: bool = True


@lru_cache
def get_settings() -> Settings:
    return Settings()
