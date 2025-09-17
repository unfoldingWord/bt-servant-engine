"""Runtime configuration via Pydantic BaseSettings.

Loads values from environment (and optional .env) with typed fields.
"""
import os
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    """Typed configuration model sourced from environment variables."""
    # Model settings
    model_config = SettingsConfigDict(env_file=".env")

    OPENAI_API_KEY: str = Field(...)
    META_VERIFY_TOKEN: str = Field(...)
    META_WHATSAPP_TOKEN: str = Field(...)
    META_PHONE_NUMBER_ID: str = Field(...)
    META_APP_SECRET: str = Field(...)
    FACEBOOK_USER_AGENT: str = Field(...)
    IN_META_SANDBOX_MODE: bool = Field(default=False)
    META_SANDBOX_PHONE_NUMBER: str = Field(default="11111111")
    MESSAGE_AGE_CUTOFF_IN_SECONDS: int = Field(default=3600)
    BASE_URL: str = Field(...)
    BT_SERVANT_LOG_LEVEL: str = Field(default="info")
    MAX_META_TEXT_LENGTH: int = Field(default=4096)
    # Max verses to include in get-translation-helps context to control token usage
    TRANSLATION_HELPS_VERSE_LIMIT: int = Field(default=5)
    # Max verses allowed for retrieve-scripture (prevents huge selections like an entire book)
    RETRIEVE_SCRIPTURE_VERSE_LIMIT: int = Field(default=120)
    # Max verses allowed for translate-scripture (avoid very large translations)
    TRANSLATE_SCRIPTURE_VERSE_LIMIT: int = Field(default=120)
    # Admin API token for protecting CRUD endpoints
    ADMIN_API_TOKEN: str | None = Field(default=None)
    # Dedicated token for health check authentication
    HEALTHCHECK_API_TOKEN: str | None = Field(default=None)
    # Enable admin auth for protected endpoints (default False for local/dev tests)
    ENABLE_ADMIN_AUTH: bool = Field(default=True)
    # Tuning knob for LLM creativity/agency (normal | low)
    AGENTIC_STRENGTH: Literal["normal", "low"] = Field(default="normal")

    # Optional with default value
    DATA_DIR: Path = Field(default=Path("/data"))
    # Default OpenAI pricing JSON to enable cost accounting even without .env override
    OPENAI_PRICING_JSON: str = Field(
        default=(
            '{'
            '"gpt-4o": {"input_per_million": 2.5, '
            '"output_per_million": 10.0, "cached_input": 1.25}, '
            '"gpt-4o-mini": {"input_per_million": 0.15, '
            '"output_per_million": 0.6}, '
            '"gpt-4o-transcribe": {"input_per_million": 2.5, '
            '"output_per_million": 10.0, '
            '"audio_input_per_million": 6.0}, '
            '"gpt-4o-mini-tts": {"input_per_million": 0.6, '
            '"audio_output_per_million": 12.0}'
            '}'
        )
    )


# Create a single instance to import elsewhere
# mypy cannot infer environment-based initialization for BaseSettings
config = Config()  # type: ignore[call-arg]

# Ensure utils.pricing (which reads from environment) can see a default when .env omits it
os.environ.setdefault("OPENAI_PRICING_JSON", config.OPENAI_PRICING_JSON)
# Surface OpenAI credentials for libraries that only check raw environment vars
os.environ.setdefault("OPENAI_API_KEY", config.OPENAI_API_KEY)
