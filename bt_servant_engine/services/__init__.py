"""Application service layer scaffolding for intent handling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING


if TYPE_CHECKING:  # pragma: no cover - type narrowing only
    from .intent_router import IntentRouter


@dataclass(slots=True)
class ServiceContainer:
    """Aggregate of application-level services available to handlers."""

    intent_router: Optional["IntentRouter"] = None


def build_default_services() -> ServiceContainer:
    """Return a service container with the default intent router wiring."""

    from brain import IntentType  # pylint: disable=import-outside-toplevel

    from .intent_router import IntentRouter  # pylint: disable=import-outside-toplevel
    from .intents.converse import (  # pylint: disable=import-outside-toplevel
        handle_converse_intent,
    )

    container = ServiceContainer()
    container.intent_router = IntentRouter(
        {IntentType.CONVERSE_WITH_BT_SERVANT: handle_converse_intent}
    )
    return container


__all__ = ["ServiceContainer", "build_default_services"]
