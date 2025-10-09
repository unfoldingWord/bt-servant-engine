"""Handlers for conversational fallback intents."""

from __future__ import annotations

from typing import TYPE_CHECKING

from bt_servant_engine.core.intents import IntentType

from ..intent_router import IntentRequest, IntentResponse

if TYPE_CHECKING:  # pragma: no cover - static typing aid
    from .. import ServiceContainer


async def handle_converse_intent(
    request: IntentRequest, services: "ServiceContainer"
) -> IntentResponse:
    """Return a simple acknowledgement for conversational intents."""

    _ = services  # Placeholder for future shared dependencies
    user_text = str(request.payload.get("text", "")).strip()
    if not user_text:
        message = "I'm here whenever you need assistance."
    else:
        message = f"You said: {user_text}"
    return IntentResponse(
        intent=IntentType.CONVERSE_WITH_BT_SERVANT,
        result={"text": message},
    )


__all__ = ["handle_converse_intent"]
