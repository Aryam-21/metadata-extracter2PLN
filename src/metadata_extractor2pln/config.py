from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache


def _csv(name: str, default: str = "") -> tuple[str, ...]:
    return tuple(
        item.strip() for item in os.getenv(name, default).split(",") if item.strip()
    )


def _bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    return (
        default
        if value is None
        else value.strip().lower() in {"1", "true", "yes", "on"}
    )


@dataclass(frozen=True)
class Settings:
    environment: str = "production"
    api_keys: tuple[str, ...] = ()
    allowed_hosts: tuple[str, ...] = ("localhost", "127.0.0.1")
    model_provider: str = "gemini"
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-2.5-flash"
    asi_api_key: str | None = None
    asi_model: str = "asi1-mini"
    asi_base_url: str = "https://api.asi1.ai/v1"
    model_timeout_seconds: float = 45.0
    request_timeout_seconds: float = 120.0
    max_request_bytes: int = 2_097_152
    max_concurrent_requests: int = 4
    rate_limit_per_minute: int = 60
    expose_docs: bool = False

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            environment=os.getenv("METADATA_ENVIRONMENT", "production"),
            api_keys=_csv("METADATA_API_KEYS"),
            allowed_hosts=_csv("METADATA_ALLOWED_HOSTS", "localhost,127.0.0.1"),
            model_provider=os.getenv("METADATA_MODEL_PROVIDER", "gemini").strip().lower(),
            gemini_api_key=os.getenv("GEMINI_API_KEY") or None,
            gemini_model=os.getenv("METADATA_GEMINI_MODEL", "gemini-2.5-flash"),
            asi_api_key=os.getenv("ASI_ONE_API_KEY") or None,
            asi_model=os.getenv("METADATA_ASI_MODEL", "asi1-mini"),
            asi_base_url=os.getenv("METADATA_ASI_BASE_URL", "https://api.asi1.ai/v1"),
            model_timeout_seconds=float(
                os.getenv("METADATA_MODEL_TIMEOUT_SECONDS", "45")
            ),
            request_timeout_seconds=float(
                os.getenv("METADATA_REQUEST_TIMEOUT_SECONDS", "120")
            ),
            max_request_bytes=int(os.getenv("METADATA_MAX_REQUEST_BYTES", "2097152")),
            max_concurrent_requests=int(
                os.getenv("METADATA_MAX_CONCURRENT_REQUESTS", "4")
            ),
            rate_limit_per_minute=int(
                os.getenv("METADATA_RATE_LIMIT_PER_MINUTE", "60")
            ),
            expose_docs=_bool("METADATA_EXPOSE_DOCS", False),
        )

    def validate(self) -> None:
        if self.environment != "test" and not self.api_keys:
            raise ValueError(
                "METADATA_API_KEYS must contain at least one owner:secret entry"
            )
        for entry in self.api_keys:
            owner, separator, secret = entry.partition(":")
            if not separator or not owner.replace("-", "").replace("_", "").isalnum():
                raise ValueError("API keys must use owner-id:secret format")
            if self.environment != "test" and len(secret) < 32:
                raise ValueError("API key secrets must contain at least 32 characters")
        if not self.allowed_hosts:
            raise ValueError("METADATA_ALLOWED_HOSTS cannot be empty")
        if self.model_provider not in {"gemini", "asi"}:
            raise ValueError("METADATA_MODEL_PROVIDER must be gemini or asi")
        if self.max_concurrent_requests < 1 or self.rate_limit_per_minute < 1:
            raise ValueError("concurrency and rate limits must be positive")


@lru_cache
def get_settings() -> Settings:
    settings = Settings.from_env()
    settings.validate()
    return settings
