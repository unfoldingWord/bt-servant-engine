"""Shared FastAPI dependencies for token validation and service access."""

import hmac
from typing import Annotated, Optional

from fastapi import Depends, Header, HTTPException, status

from bt_servant_engine.core.config import config
from bt_servant_engine.services import ServiceContainer, runtime
from bt_servant_engine.services.admin import AdminDatastoreService


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
    "get_service_container",
    "require_admin_token",
    "require_healthcheck_token",
]
