"""Application settings, loaded from environment variables and an optional ``.env`` file."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_BOOLS = {"true": True, "1": True, "yes": True, "false": False, "0": False, "no": False}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(".env", "../.env"),
        extra="ignore",
        case_sensitive=False,
    )

    # ArangoDB connection. On the Arango Platform the endpoint and CA are injected and each
    # request carries the signed-in user's token, so only the database is chosen per request
    # (X-Arango-Database). Off the platform the username/password below are exchanged for a JWT.
    arango_endpoint: str = "http://localhost:8529"
    arango_database: str = "_system"
    arango_username: str = "root"
    arango_password: SecretStr = SecretStr("")
    arango_verify_ssl: bool | str = Field(
        True, description="true/false, or the path of a CA bundle to verify against."
    )
    arango_request_timeout: int = 120

    # Arango Platform integration sidecar (injected). Mints tokens for work that outlives a
    # request: load jobs, watchers, and the watchers started at boot (as ``arango_service_user``).
    integration_http_address: str | None = Field(
        None,
        validation_alias=AliasChoices("INTEGRATION_HTTP_ADDRESS_FULL", "INTEGRATION_HTTP_ADDRESS"),
    )
    arango_service_user: str = "root"
    token_lifetime_s: int = 3600

    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:5173"])
    static_dir: Path | None = None
    max_upload_mb: int = 1024
    upload_dir: Path | None = None

    # Remote sources (POST /load/url). Empty allow-list = any host.
    allow_url_sources: bool = True
    url_allowed_hosts: list[str] = Field(default_factory=list)
    #: Sanity cap on a URL load saved to disk for the `long_input: error` pre-scan (all URLs).
    max_url_download_mb: int = 500

    # Embedding defaults
    embed_concurrency: int = 4
    default_batch_size: int = 64
    presets_file: Path | None = None

    # Collection embedding profiles
    profiles_collection: str = "embedding_loader_profiles"
    start_watchers_on_startup: bool = True

    @field_validator("arango_verify_ssl", mode="before")
    @classmethod
    def _parse_verify(cls, value: object) -> object:
        if isinstance(value, str) and value.strip().lower() in _BOOLS:
            return _BOOLS[value.strip().lower()]
        return value

    @property
    def on_platform(self) -> bool:
        """Running on the Arango Platform with the integration sidecar injected."""
        return bool(self.integration_http_address)

    @model_validator(mode="after")
    def _use_platform_endpoint(self) -> Settings:
        """On the Arango Platform (BYOC) the operator injects the coordinator's in-cluster
        address as ``ARANGO_DEPLOYMENT_ENDPOINT`` and its CA as ``ARANGO_DEPLOYMENT_CA``.
        Use them unless ``ARANGO_ENDPOINT`` / ``ARANGO_VERIFY_SSL`` are set."""
        injected = os.getenv("ARANGO_DEPLOYMENT_ENDPOINT", "").strip()
        if injected and "arango_endpoint" not in self.model_fields_set:
            self.arango_endpoint = injected.rstrip("/")
            if "arango_verify_ssl" not in self.model_fields_set:
                ca = os.getenv("ARANGO_DEPLOYMENT_CA", "").strip()
                self.arango_verify_ssl = ca if ca and Path(ca).is_file() else False
        if self.integration_http_address and "://" not in self.integration_http_address:
            self.integration_http_address = f"http://{self.integration_http_address}"
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
