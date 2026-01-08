"""User preferences API endpoints."""

from __future__ import annotations

from typing import Literal, cast

from fastapi import APIRouter, Depends, HTTPException, Request, status

from bt_servant_engine.apps.api.dependencies import require_client_api_key
from bt_servant_engine.core.api_key_models import APIKey
from bt_servant_engine.core.api_models import UserPreferences
from bt_servant_engine.core.config import config
from bt_servant_engine.core.logging import get_logger
from bt_servant_engine.core.ports import UserStatePort
from bt_servant_engine.services import ServiceContainer

router = APIRouter(prefix="/api/v1/users", tags=["users"])
logger = get_logger(__name__)

# Type alias matching UserPreferences.agentic_strength
AgenticStrength = Literal["normal", "low", "very_low"] | None


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


@router.get("/{user_id}/preferences", response_model=UserPreferences)
async def get_user_preferences(
    request: Request,
    user_id: str,
    api_key: APIKey = Depends(require_client_api_key),  # noqa: ARG001
) -> UserPreferences:
    """Get user preferences."""
    services = _get_services(request)
    user_state = _require_user_state(services)

    return UserPreferences(
        response_language=user_state.get_response_language(user_id=user_id),
        agentic_strength=cast(AgenticStrength, user_state.get_agentic_strength(user_id=user_id)),
        dev_agentic_mcp=user_state.get_dev_agentic_mcp(user_id=user_id),
    )


@router.put("/{user_id}/preferences", response_model=UserPreferences)
async def update_user_preferences(
    request: Request,
    user_id: str,
    preferences: UserPreferences,
    api_key: APIKey = Depends(require_client_api_key),  # noqa: ARG001
) -> UserPreferences:
    """Update user preferences. Only provided fields are updated."""
    services = _get_services(request)
    user_state = _require_user_state(services)

    # Update only provided fields
    if preferences.response_language is not None:
        user_state.set_response_language(user_id=user_id, language=preferences.response_language)

    if preferences.agentic_strength is not None:
        user_state.set_agentic_strength(user_id=user_id, strength=preferences.agentic_strength)

    if preferences.dev_agentic_mcp is not None:
        user_state.set_dev_agentic_mcp(user_id=user_id, enabled=preferences.dev_agentic_mcp)

    # Return current state
    return UserPreferences(
        response_language=user_state.get_response_language(user_id=user_id),
        agentic_strength=cast(AgenticStrength, user_state.get_agentic_strength(user_id=user_id)),
        dev_agentic_mcp=user_state.get_dev_agentic_mcp(user_id=user_id),
    )
