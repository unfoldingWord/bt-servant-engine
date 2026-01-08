"""Shared FastAPI dependencies for token validation and service access."""

import hmac
from datetime import datetime, timezone
from typing import Annotated, Optional

from fastapi import Depends, Header, HTTPException, status

from bt_servant_engine.core.api_key_models import APIKey, Environment
from bt_servant_engine.core.config import config
from bt_servant_engine.services import ServiceContainer, runtime
from bt_servant_engine.services.admin import AdminDatastoreService
from bt_servant_engine.services.api_keys import APIKeyService

# Global API key service (initialized at startup)
_api_key_service: APIKeyService | None = None


def set_api_key_service(service: APIKeyService) -> None:
    """Set the global API key service (called at app startup)."""
    global _api_key_service  # noqa: PLW0603
    _api_key_service = service


def get_api_key_service() -> APIKeyService | None:
    """Get the API key service, or None if not initialized."""
    return _api_key_service


async def _validate_token(
    *,
    expected: Optional[str],
    authorization: Annotated[Optional[str], Header(alias="Authorization")] = None,
    x_admin_token: Annotated[Optional[str], Header(alias="X-Admin-Token")] = None,
) -> None:
    """Shared helper to validate bearer/X-Admin-Token headers."""
    if not expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token not configured",
            headers={"WWW-Authenticate": "Bearer"},
        )

    provided = None
    if authorization and authorization.lower().startswith("bearer "):
        provided = authorization.split(" ", 1)[1].strip()
    elif x_admin_token:
        provided = x_admin_token.strip()

    if not provided:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not hmac.compare_digest(provided, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def require_admin_token(
    authorization: Annotated[Optional[str], Header(alias="Authorization")] = None,
    x_admin_token: Annotated[Optional[str], Header(alias="X-Admin-Token")] = None,
) -> None:
    """Simple admin token guard for non-webhook endpoints."""
    if not config.ENABLE_ADMIN_AUTH:
        return
    await _validate_token(
        expected=config.ADMIN_API_TOKEN,
        authorization=authorization,
        x_admin_token=x_admin_token,
    )


async def require_healthcheck_token(
    authorization: Annotated[Optional[str], Header(alias="Authorization")] = None,
    x_admin_token: Annotated[Optional[str], Header(alias="X-Admin-Token")] = None,
) -> None:
    """Token guard specifically for the health check endpoint."""
    if not config.ENABLE_ADMIN_AUTH:
        return
    await _validate_token(
        expected=config.HEALTHCHECK_API_TOKEN,
        authorization=authorization,
        x_admin_token=x_admin_token,
    )


def _is_client_auth_enabled() -> bool:
    """Check if client API key auth is enabled.

    Uses ENABLE_CLIENT_API_KEY_AUTH if set, otherwise falls back to ENABLE_ADMIN_AUTH.
    """
    if config.ENABLE_CLIENT_API_KEY_AUTH is not None:
        return config.ENABLE_CLIENT_API_KEY_AUTH
    return config.ENABLE_ADMIN_AUTH


async def require_client_api_key(
    authorization: Annotated[Optional[str], Header(alias="Authorization")] = None,
) -> APIKey:
    """Validate client API key and return key metadata.

    For use with chat/users endpoints. Client keys are separate from ADMIN_API_TOKEN.
    Falls back to ADMIN_API_TOKEN if no client key system is configured (backward compat).

    Set ENABLE_CLIENT_API_KEY_AUTH=false to disable client API key requirement.
    """
    # If client auth is disabled, return a dummy key
    if not _is_client_auth_enabled():
        return APIKey(
            id="auth-disabled",
            key_prefix="disabled",
            name="Auth Disabled",
            environment=Environment.DEV,
            created_at=datetime.now(timezone.utc),
        )

    # If client key system is not initialized, fall back to admin token
    if _api_key_service is None:
        await require_admin_token(authorization=authorization)
        return APIKey(
            id="admin-fallback",
            key_prefix="admin",
            name="Admin Token (legacy)",
            environment=Environment.PROD,
            created_at=datetime.now(timezone.utc),
        )

    # Extract bearer token
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not authorization.lower().startswith("bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Authorization header format. Expected 'Bearer <token>'",
            headers={"WWW-Authenticate": "Bearer"},
        )

    raw_key = authorization.split(" ", 1)[1].strip()

    # Check if it's the admin token (backward compatibility)
    if config.ADMIN_API_TOKEN and hmac.compare_digest(raw_key, config.ADMIN_API_TOKEN):
        return APIKey(
            id="admin-fallback",
            key_prefix="admin",
            name="Admin Token (legacy)",
            environment=Environment.PROD,
            created_at=datetime.now(timezone.utc),
        )

    # Validate as client API key
    result = _api_key_service.validate_key(raw_key)

    if not result.valid:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=result.error or "Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )

    assert result.key is not None  # Type narrowing
    return result.key


def get_service_container() -> ServiceContainer:
    """Resolve the globally configured service container."""
    try:
        return runtime.get_services()
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Service container not configured",
        ) from exc


def get_admin_datastore_service(
    container: Annotated[ServiceContainer, Depends(get_service_container)],
) -> AdminDatastoreService:
    """Return an admin datastore service bound to the active container."""
    if container.chroma is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Chroma service is unavailable",
        )
    return AdminDatastoreService(container.chroma)


__all__ = [
    "get_admin_datastore_service",
    "get_api_key_service",
    "get_service_container",
    "require_admin_token",
    "require_client_api_key",
    "require_healthcheck_token",
    "set_api_key_service",
]
