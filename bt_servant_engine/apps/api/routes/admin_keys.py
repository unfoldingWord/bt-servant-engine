"""Administrative routes for API key management."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from bt_servant_engine.apps.api.dependencies import (
    get_api_key_service,
    require_admin_token,
)
from bt_servant_engine.core.api_key_models import APIKey

router = APIRouter(prefix="/admin/keys")

AuthDependency = Annotated[None, Depends(require_admin_token)]


class CreateAPIKeyRequest(BaseModel):
    """Request model for creating a new API key."""

    name: str = Field(..., description="Human-readable name for the key")
    environment: str = Field(
        default="prod", description="Environment: prod, staging, or dev"
    )
    rate_limit_per_minute: int = Field(default=60, ge=1, description="Rate limit")
    expires_in_days: int | None = Field(
        default=None, ge=1, description="Days until expiration (null = never)"
    )


class APIKeyInfo(BaseModel):
    """API key information (never includes the raw key)."""

    id: str
    key_prefix: str
    name: str
    environment: str
    created_at: datetime
    revoked_at: datetime | None
    last_used_at: datetime | None
    rate_limit_per_minute: int
    expires_at: datetime | None
    is_active: bool


class CreateAPIKeyResponse(BaseModel):
    """Response after creating an API key - includes the raw key shown once."""

    key: APIKeyInfo
    raw_key: str = Field(..., description="The full API key - SAVE THIS, shown once only!")


class ListAPIKeysResponse(BaseModel):
    """Response for listing API keys."""

    keys: list[APIKeyInfo]
    total: int


def _api_key_to_info(key: APIKey) -> APIKeyInfo:
    """Convert domain APIKey to APIKeyInfo response model."""
    return APIKeyInfo(
        id=key.id,
        key_prefix=key.key_prefix,
        name=key.name,
        environment=key.environment.value,
        created_at=key.created_at,
        revoked_at=key.revoked_at,
        last_used_at=key.last_used_at,
        rate_limit_per_minute=key.rate_limit_per_minute,
        expires_at=key.expires_at,
        is_active=key.is_active,
    )


@router.post("", response_model=CreateAPIKeyResponse, status_code=status.HTTP_201_CREATED)
async def create_api_key(
    request: CreateAPIKeyRequest,
    _: AuthDependency,
) -> CreateAPIKeyResponse:
    """Create a new API key.

    Returns the full key value - this is the only time it will be shown.
    The key is stored as a bcrypt hash and cannot be recovered.
    """
    service = get_api_key_service()
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="API key service not initialized",
        )

    expires_at = None
    if request.expires_in_days:
        expires_at = datetime.now(timezone.utc) + timedelta(days=request.expires_in_days)

    try:
        result = service.create_key(
            name=request.name,
            environment=request.environment,
            rate_limit_per_minute=request.rate_limit_per_minute,
            expires_at=expires_at,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    return CreateAPIKeyResponse(
        key=_api_key_to_info(result.key),
        raw_key=result.raw_key,
    )


@router.get("", response_model=ListAPIKeysResponse)
async def list_api_keys(
    _: AuthDependency,
    include_revoked: bool = False,
    environment: str | None = None,
) -> ListAPIKeysResponse:
    """List all API keys.

    By default, only active keys are shown. Use include_revoked=true to see all.
    """
    service = get_api_key_service()
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="API key service not initialized",
        )

    keys = service.list_keys(include_revoked=include_revoked, environment=environment)
    return ListAPIKeysResponse(
        keys=[_api_key_to_info(k) for k in keys],
        total=len(keys),
    )


@router.get("/{key_id}", response_model=APIKeyInfo)
async def get_api_key(
    key_id: str,
    _: AuthDependency,
) -> APIKeyInfo:
    """Get details for a specific API key by ID."""
    service = get_api_key_service()
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="API key service not initialized",
        )

    key = service.get_key_by_id(key_id)
    if key is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"API key not found: {key_id}",
        )

    return _api_key_to_info(key)


@router.delete("/{key_id}")
async def revoke_api_key(
    key_id: str,
    _: AuthDependency,
) -> JSONResponse:
    """Revoke an API key.

    Once revoked, the key can no longer be used for authentication.
    This action cannot be undone.
    """
    service = get_api_key_service()
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="API key service not initialized",
        )

    success = service.revoke_key(key_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"API key not found or already revoked: {key_id}",
        )

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={"message": "API key revoked successfully", "key_id": key_id},
    )


__all__ = ["router"]
