"""Core domain models for API key management."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field

# Key format: bts_<env>_<random>
KEY_PREFIX_LENGTH = 12  # "bts_prod_a1b" - used for fast lookup


class Environment(str, Enum):
    """Valid API key environments."""

    PROD = "prod"
    STAGING = "staging"
    DEV = "dev"


class APIKey(BaseModel):
    """Domain model for an API key (never contains the raw key)."""

    id: str = Field(..., description="UUID of the key")
    key_prefix: str = Field(..., description="First 12 chars for display/lookup")
    name: str = Field(..., description="Human-readable name")
    environment: Environment = Field(..., description="Target environment")
    created_at: datetime = Field(..., description="Creation timestamp")
    revoked_at: datetime | None = Field(
        default=None, description="Revocation timestamp if revoked"
    )
    last_used_at: datetime | None = Field(
        default=None, description="Last successful auth"
    )
    rate_limit_per_minute: int = Field(default=60, description="Rate limit")
    expires_at: datetime | None = Field(
        default=None, description="Expiration timestamp"
    )

    @property
    def is_active(self) -> bool:
        """Return True if key is not revoked and not expired."""
        if self.revoked_at is not None:
            return False
        if self.expires_at is not None and self.expires_at < datetime.now(timezone.utc):
            return False
        return True


class APIKeyCreateResult(BaseModel):
    """Result of creating a new API key - includes the raw key shown once."""

    key: APIKey = Field(..., description="The created key metadata")
    raw_key: str = Field(..., description="The full API key - SHOWN ONCE ONLY")


class APIKeyValidation(BaseModel):
    """Result of validating an API key."""

    valid: bool = Field(..., description="Whether the key is valid")
    key: APIKey | None = Field(default=None, description="Key metadata if valid")
    error: str | None = Field(default=None, description="Error message if invalid")


__all__ = [
    "Environment",
    "APIKey",
    "APIKeyCreateResult",
    "APIKeyValidation",
]
