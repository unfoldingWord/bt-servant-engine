"""User preferences API endpoints."""

from __future__ import annotations

from datetime import datetime
from typing import Literal, cast

from fastapi import APIRouter, Depends, Request

from bt_servant_engine.apps.api.dependencies import require_client_api_key
from bt_servant_engine.core.api_key_models import APIKey
from bt_servant_engine.adapters.user_state import get_all_user_ids
from bt_servant_engine.core.api_models import (
    ChatHistoryEntry,
    ChatHistoryResponse,
    UserPreferences,
)
from bt_servant_engine.core.config import config  # noqa: F401 - used by test fixtures
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
    api_key: APIKey = Depends(require_client_api_key),  # noqa: ARG001, B008
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
    api_key: APIKey = Depends(require_client_api_key),  # noqa: ARG001, B008
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


@router.get("/debug/all-ids", response_model=list[str])
async def get_all_users(
    api_key: APIKey = Depends(require_client_api_key),  # noqa: ARG001, B008
) -> list[str]:
    """Get all user IDs in the database (for debugging)."""
    return get_all_user_ids()


def _parse_created_at(value: str | None) -> datetime | None:
    """Parse ISO format timestamp, returning None for missing/invalid values."""
    if value is None:
        return None
    try:
        return datetime.fromisoformat(value)
    except (ValueError, TypeError):
        return None


@router.get("/{user_id}/history", response_model=ChatHistoryResponse)
async def get_user_history(
    request: Request,
    user_id: str,
    limit: int = 50,
    offset: int = 0,
    api_key: APIKey = Depends(require_client_api_key),  # noqa: ARG001, B008
) -> ChatHistoryResponse:
    """Get user's chat history with pagination.

    Args:
        user_id: User identifier
        limit: Maximum entries to return (default 50, max 100)
        offset: Number of entries to skip (for pagination)

    Returns:
        Chat history entries with timestamps (newest entries first in response)
    """
    services = _get_services(request)
    user_state = _require_user_state(services)

    # Clamp limit to reasonable bounds
    limit = max(1, min(limit, 100))
    offset = max(0, offset)

    # Get full history (stored oldest to newest)
    full_history = user_state.get_chat_history(user_id=user_id)
    total_count = len(full_history)

    # Reverse for newest-first, then apply pagination
    reversed_history = list(reversed(full_history))
    paginated = reversed_history[offset : offset + limit]

    # Convert to response model
    entries = [
        ChatHistoryEntry(
            user_message=entry.get("user_message", ""),
            assistant_response=entry.get("assistant_response", ""),
            created_at=_parse_created_at(entry.get("created_at")),
        )
        for entry in paginated
    ]

    return ChatHistoryResponse(
        user_id=user_id,
        entries=entries,
        total_count=total_count,
        limit=limit,
        offset=offset,
    )
