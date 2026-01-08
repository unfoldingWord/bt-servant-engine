"""API key management service - business logic layer."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from bt_servant_engine.core.api_key_models import (
    KEY_PREFIX_LENGTH,
    APIKey,
    APIKeyCreateResult,
    APIKeyValidation,
)

if TYPE_CHECKING:
    from bt_servant_engine.core.ports import APIKeyPort


class APIKeyService:
    """Business logic for API key management."""

    def __init__(self, port: APIKeyPort) -> None:
        """Initialize the service with an API key port."""
        self._port = port

    def create_key(
        self,
        name: str,
        environment: str = "prod",
        rate_limit_per_minute: int = 60,
        expires_at: datetime | None = None,
    ) -> APIKeyCreateResult:
        """Create a new API key.

        Args:
            name: Human-readable name for the key.
            environment: Target environment (prod, staging, dev).
            rate_limit_per_minute: Rate limit for this key.
            expires_at: Optional expiration timestamp.

        Returns:
            APIKeyCreateResult containing the key metadata and raw key.

        Raises:
            ValueError: If name is empty or environment is invalid.
        """
        if not name or not name.strip():
            raise ValueError("Key name cannot be empty")
        if environment not in ("prod", "staging", "dev"):
            raise ValueError(f"Invalid environment: {environment}")
        if rate_limit_per_minute < 1:
            raise ValueError("Rate limit must be at least 1")

        key, raw_key = self._port.create_key(
            name=name.strip(),
            environment=environment,
            rate_limit_per_minute=rate_limit_per_minute,
            expires_at=expires_at,
        )
        return APIKeyCreateResult(key=key, raw_key=raw_key)

    def validate_key(self, raw_key: str | None) -> APIKeyValidation:
        """Validate a raw API key.

        Args:
            raw_key: The raw API key to validate.

        Returns:
            APIKeyValidation with validation result and key metadata if valid.
        """
        if not raw_key:
            return APIKeyValidation(valid=False, error="No API key provided")

        key = self._port.validate_key(raw_key)
        if key is None:
            return APIKeyValidation(valid=False, error="Invalid API key")

        if not key.is_active:
            reason = "revoked" if key.revoked_at else "expired"
            return APIKeyValidation(valid=False, error=f"API key is {reason}")

        # Update last used timestamp
        self._port.update_last_used(key.id)

        return APIKeyValidation(valid=True, key=key)

    def get_key_by_id(self, key_id: str) -> APIKey | None:
        """Get key metadata by ID."""
        return self._port.get_key_by_id(key_id)

    def list_keys(
        self,
        include_revoked: bool = False,
        environment: str | None = None,
    ) -> list[APIKey]:
        """List API keys.

        Args:
            include_revoked: Include revoked keys in the listing.
            environment: Filter by environment (prod, staging, dev).

        Returns:
            List of API key metadata.
        """
        return self._port.list_keys(
            include_revoked=include_revoked, environment=environment
        )

    def revoke_key(self, key_id: str) -> bool:
        """Revoke an API key by ID.

        Args:
            key_id: The UUID of the key to revoke.

        Returns:
            True if the key was revoked, False if not found or already revoked.
        """
        return self._port.revoke_key(key_id)

    def revoke_key_by_prefix(self, prefix: str) -> bool:
        """Revoke a key by its prefix (for CLI convenience).

        Args:
            prefix: The key prefix (first 12 chars) or full key.

        Returns:
            True if the key was revoked, False if not found.
        """
        # Handle full key or prefix
        lookup_prefix = prefix[:KEY_PREFIX_LENGTH] if len(prefix) > KEY_PREFIX_LENGTH else prefix
        key = self._port.get_key_by_prefix(lookup_prefix)
        if key is None:
            return False
        return self._port.revoke_key(key.id)


__all__ = ["APIKeyService"]
