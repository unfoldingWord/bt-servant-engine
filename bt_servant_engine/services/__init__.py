"""Application service layer scaffolding for intent handling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

from bt_servant_engine.core.ports import ChromaPort, MessagingPort, UserStatePort

if TYPE_CHECKING:  # pragma: no cover - type narrowing only
    from .intent_router import IntentRouter


@dataclass(slots=True)
class ServiceContainer:
    """Aggregate of application-level services available to handlers."""

    chroma: Optional[ChromaPort] = None
    user_state: Optional[UserStatePort] = None
    messaging: Optional[MessagingPort] = None
    intent_router: Optional["IntentRouter"] = None


def build_default_services(
    *,
    chroma_port: Optional[ChromaPort] = None,
    user_state_port: Optional[UserStatePort] = None,
    messaging_port: Optional[MessagingPort] = None,
) -> ServiceContainer:
    """Return a service container with the default intent router wiring."""

    from brain import IntentType  # pylint: disable=import-outside-toplevel

    from .intent_router import IntentRouter  # pylint: disable=import-outside-toplevel
    from .intents.converse import (  # pylint: disable=import-outside-toplevel
        handle_converse_intent,
    )

    intent_router = IntentRouter({IntentType.CONVERSE_WITH_BT_SERVANT: handle_converse_intent})
    return ServiceContainer(
        chroma=chroma_port,
        user_state=user_state_port,
        messaging=messaging_port,
        intent_router=intent_router,
    )


__all__ = ["ServiceContainer", "build_default_services"]
