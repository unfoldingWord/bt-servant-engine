"""User preferences API endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Header, HTTPException, Request, status

from bt_servant_engine.core.api_models import UserPreferences
from bt_servant_engine.core.config import config
from bt_servant_engine.core.logging import get_logger
from bt_servant_engine.core.ports import UserStatePort
from bt_servant_engine.services import ServiceContainer

router = APIRouter(prefix="/api/v1/users", tags=["users"])
logger = get_logger(__name__)

# Expected number of parts in "Bearer <token>" authorization header
_BEARER_AUTH_PARTS = 2


def _get_services(request: Request) -> ServiceContainer:
    """Retrieve the service container from app state."""
    services = getattr(request.app.state, "services", None)
    if not isinstance(services, ServiceContainer):
        raise RuntimeError("Service container is not configured on app.state.")
    return services


def _require_user_state(services: ServiceContainer) -> UserStatePort:
    """Get UserStatePort from services or raise."""
    user_state = services.user_state
    if user_state is None:
        raise RuntimeError("UserStatePort has not been configured.")
    return user_state


async def _verify_api_key(authorization: str | None = Header(default=None)) -> None:
    """Verify the API key from the Authorization header."""
    if not config.ADMIN_API_TOKEN:
        # If no token is configured, allow all requests (dev mode)
        return

    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header",
        )

    parts = authorization.split()
    if len(parts) != _BEARER_AUTH_PARTS or parts[0].lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Authorization header format. Expected 'Bearer <token>'",
        )

    if parts[1] != config.ADMIN_API_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )


@router.get("/{user_id}/preferences", response_model=UserPreferences)
async def get_user_preferences(
    request: Request,
    user_id: str,
    _: None = Depends(_verify_api_key),
) -> UserPreferences:
    """Get user preferences."""
    services = _get_services(request)
    user_state = _require_user_state(services)

    return UserPreferences(
        response_language=user_state.get_response_language(user_id=user_id),
        agentic_strength=user_state.get_agentic_strength(user_id=user_id),
        dev_agentic_mcp=user_state.get_dev_agentic_mcp(user_id=user_id),
    )


@router.put("/{user_id}/preferences", response_model=UserPreferences)
async def update_user_preferences(
    request: Request,
    user_id: str,
    preferences: UserPreferences,
    _: None = Depends(_verify_api_key),
) -> UserPreferences:
    """Update user preferences. Only provided fields are updated."""
    services = _get_services(request)
    user_state = _require_user_state(services)

    # Update only provided fields
    if preferences.response_language is not None:
        user_state.set_response_language(
            user_id=user_id, language=preferences.response_language
        )

    if preferences.agentic_strength is not None:
        user_state.set_agentic_strength(
            user_id=user_id, strength=preferences.agentic_strength
        )

    if preferences.dev_agentic_mcp is not None:
        user_state.set_dev_agentic_mcp(user_id=user_id, enabled=preferences.dev_agentic_mcp)

    # Return current state
    return UserPreferences(
        response_language=user_state.get_response_language(user_id=user_id),
        agentic_strength=user_state.get_agentic_strength(user_id=user_id),
        dev_agentic_mcp=user_state.get_dev_agentic_mcp(user_id=user_id),
    )
