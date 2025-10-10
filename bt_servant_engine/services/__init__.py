"""Application service layer scaffolding for intent handling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

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

    from .intent_router import IntentRouter  # pylint: disable=import-outside-toplevel

    intent_router = IntentRouter()
    return ServiceContainer(
        chroma=chroma_port,
        user_state=user_state_port,
        messaging=messaging_port,
        intent_router=intent_router,
    )


__all__ = ["ServiceContainer", "build_default_services"]
