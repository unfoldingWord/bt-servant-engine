"""Intent router and supporting request/response models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Mapping, MutableMapping, Sequence

from brain import IntentType

if TYPE_CHECKING:  # pragma: no cover - import for static analysis only
    from . import ServiceContainer


IntentPayload = dict[str, Any]


@dataclass(slots=True)
class IntentRequest:
    """Normalized intent invocation produced by upstream classifiers."""

    intent: IntentType
    payload: IntentPayload


@dataclass(slots=True)
class IntentResponse:
    """Uniform handler response structure for downstream translation."""

    intent: IntentType
    result: Any


IntentHandler = Callable[[IntentRequest, "ServiceContainer"], Awaitable[IntentResponse]]


class IntentRouterError(RuntimeError):
    """Base error for router failures."""


class IntentHandlerNotFoundError(IntentRouterError):
    """Raised when no handler is registered for the requested intent."""


class IntentRouter:
    """Dispatch intents to registered handlers."""

    def __init__(self, handlers: Mapping[IntentType, IntentHandler] | None = None) -> None:
        self._handlers: MutableMapping[IntentType, IntentHandler] = dict(handlers or {})

    def register(self, intent: IntentType, handler: IntentHandler) -> None:
        """Register or replace a handler for ``intent``."""

        self._handlers[intent] = handler

    def unregister(self, intent: IntentType) -> None:
        """Remove a handler if present."""

        self._handlers.pop(intent, None)

    async def dispatch(
        self, request: IntentRequest, services: "ServiceContainer"
    ) -> IntentResponse:
        """Invoke the handler for ``request.intent`` with the provided services."""

        try:
            handler = self._handlers[request.intent]
        except KeyError as exc:  # pragma: no cover - defensive guard
            raise IntentHandlerNotFoundError(
                f"No handler registered for intent {request.intent}"
            ) from exc
        return await handler(request, services)

    async def dispatch_many(
        self, requests: Sequence[IntentRequest], services: "ServiceContainer"
    ) -> list[IntentResponse]:
        """Dispatch a collection of requests sequentially and collect responses."""

        return [await self.dispatch(req, services) for req in requests]

    def handlers(self) -> Mapping[IntentType, IntentHandler]:
        """Return a shallow copy of the current intent handler registry."""

        return dict(self._handlers)


__all__ = [
    "IntentRouter",
    "IntentRouterError",
    "IntentHandlerNotFoundError",
    "IntentHandler",
    "IntentRequest",
    "IntentResponse",
]
