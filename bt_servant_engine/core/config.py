"""Application configuration loaded via Pydantic settings."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed configuration sourced from environment variables."""

    model_config = SettingsConfigDict(env_file=".env")

    OPENAI_API_KEY: str = Field(...)
    META_VERIFY_TOKEN: str = Field(...)
    META_WHATSAPP_TOKEN: str = Field(...)
    META_PHONE_NUMBER_ID: str = Field(...)
    META_APP_SECRET: str = Field(...)
    LOG_PSEUDONYM_SECRET: str = Field(...)
    FACEBOOK_USER_AGENT: str = Field(...)
    IN_META_SANDBOX_MODE: bool = Field(default=False)
    META_SANDBOX_PHONE_NUMBER: str = Field(default="11111111")
    MESSAGE_AGE_CUTOFF_IN_SECONDS: int = Field(default=3600)
    BASE_URL: str = Field(...)
    BT_SERVANT_LOG_LEVEL: str = Field(default="info")
    BT_SERVANT_LOG_DIR: Path | None = Field(default=None)
    MAX_META_TEXT_LENGTH: int = Field(default=4096)
    TRANSLATION_HELPS_VERSE_LIMIT: int = Field(default=5)
    RETRIEVE_SCRIPTURE_VERSE_LIMIT: int = Field(default=120)
    TRANSLATE_SCRIPTURE_VERSE_LIMIT: int = Field(default=120)
    ADMIN_API_TOKEN: str | None = Field(default=None)
    HEALTHCHECK_API_TOKEN: str | None = Field(default=None)
    ENABLE_ADMIN_AUTH: bool = Field(default=True)
    AGENTIC_STRENGTH: Literal["normal", "low", "very_low"] = Field(default="low")

    # Progress messaging configuration
    PROGRESS_MESSAGES_ENABLED: bool = Field(default=True)
    PROGRESS_MESSAGE_MIN_INTERVAL: float = Field(default=3.0)
    PROGRESS_MESSAGE_EMOJI: str = Field(default="⏳")
    PROGRESS_MESSAGE_EMOJI_OVERRIDES: dict[str, str] = Field(default_factory=dict)

    # Cache configuration
    CACHE_ENABLED: bool = Field(default=True)
    CACHE_BACKEND: Literal["disk", "memory"] = Field(default="disk")
    CACHE_DISK_MAX_BYTES: int = Field(default=500 * 1024 * 1024)  # 500MB
    CACHE_DEFAULT_TTL_SECONDS: int = Field(default=-1)
    CACHE_SELECTION_ENABLED: bool = Field(default=False)
    CACHE_SELECTION_TTL_SECONDS: int = Field(default=-1)
    CACHE_SELECTION_MAX_ENTRIES: int = Field(default=5000)
    CACHE_SUMMARY_ENABLED: bool = Field(default=True)
    CACHE_SUMMARY_TTL_SECONDS: int = Field(default=-1)
    CACHE_SUMMARY_MAX_ENTRIES: int = Field(default=1500)
    CACHE_KEYWORDS_ENABLED: bool = Field(default=True)
    CACHE_KEYWORDS_TTL_SECONDS: int = Field(default=-1)
    CACHE_KEYWORDS_MAX_ENTRIES: int = Field(default=3000)
    CACHE_TRANSLATION_HELPS_ENABLED: bool = Field(default=True)
    CACHE_TRANSLATION_HELPS_TTL_SECONDS: int = Field(default=-1)
    CACHE_TRANSLATION_HELPS_MAX_ENTRIES: int = Field(default=1000)
    CACHE_RAG_VECTOR_ENABLED: bool = Field(default=True)
    CACHE_RAG_VECTOR_TTL_SECONDS: int = Field(default=-1)
    CACHE_RAG_VECTOR_MAX_ENTRIES: int = Field(default=3000)
    CACHE_RAG_FINAL_ENABLED: bool = Field(default=True)
    CACHE_RAG_FINAL_TTL_SECONDS: int = Field(default=-1)
    CACHE_RAG_FINAL_MAX_ENTRIES: int = Field(default=1500)

    DATA_DIR: Path = Field(default=Path("/data"))
    OPENAI_PRICING_JSON: str = Field(
        default=(
            "{"
            '"gpt-4o": {"input_per_million": 2.5, '
            '"output_per_million": 10.0, "cached_input": 1.25}, '
            '"gpt-4o-mini": {"input_per_million": 0.15, '
            '"output_per_million": 0.6}, '
            '"gpt-4o-transcribe": {"input_per_million": 2.5, '
            '"output_per_million": 10.0, '
            '"audio_input_per_million": 6.0}, '
            '"gpt-4o-mini-tts": {"input_per_million": 0.6, '
            '"audio_output_per_million": 12.0}'
            "}"
        )
    )


settings = Settings()  # type: ignore[call-arg]
config = settings  # Alias for backward compatibility

os.environ.setdefault("OPENAI_PRICING_JSON", settings.OPENAI_PRICING_JSON)
os.environ.setdefault("OPENAI_API_KEY", settings.OPENAI_API_KEY)


__all__ = ["Settings", "settings", "config"]
