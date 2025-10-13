"""Unit tests for the intent router scaffolding."""

from __future__ import annotations

import asyncio

import pytest

from bt_servant_engine.core.intents import IntentType
from bt_servant_engine.services import ServiceContainer, build_default_services
from bt_servant_engine.services.intent_router import (
    IntentHandlerNotFoundError,
    IntentRequest,
    IntentResponse,
    IntentRouter,
)


async def echo_handler(request: IntentRequest, services: ServiceContainer) -> IntentResponse:
    """Simple echo handler used for router tests."""

    _ = services
    payload_text = str(request.payload.get("text", "")).strip()
    message = (
        f"You said: {payload_text}" if payload_text else "I'm here whenever you need assistance."
    )
    return IntentResponse(intent=request.intent, result={"text": message})


def test_dispatch_invokes_registered_handler():
    """Router calls the registered handler and returns its response."""
    router = IntentRouter({IntentType.CONVERSE_WITH_BT_SERVANT: echo_handler})
    container = ServiceContainer(intent_router=router)
    request = IntentRequest(
        intent=IntentType.CONVERSE_WITH_BT_SERVANT,
        payload={"text": "Hello there"},
    )

    response = asyncio.run(router.dispatch(request, container))

    assert isinstance(response, IntentResponse)
    assert response.intent is IntentType.CONVERSE_WITH_BT_SERVANT
    assert "Hello there" in response.result["text"]


def test_dispatch_unknown_intent_raises():
    """Router raises when no handler matches the requested intent."""
    router = IntentRouter()
    container = ServiceContainer(intent_router=router)
    request = IntentRequest(intent=IntentType.GET_PASSAGE_SUMMARY, payload={})

    with pytest.raises(IntentHandlerNotFoundError):
        asyncio.run(router.dispatch(request, container))


def test_dispatch_many_preserves_order():
    """Sequential dispatch preserves ordering of responses."""
    router = IntentRouter({IntentType.CONVERSE_WITH_BT_SERVANT: echo_handler})
    container = ServiceContainer(intent_router=router)
    requests = [
        IntentRequest(intent=IntentType.CONVERSE_WITH_BT_SERVANT, payload={"text": "First"}),
        IntentRequest(intent=IntentType.CONVERSE_WITH_BT_SERVANT, payload={"text": "Second"}),
    ]

    responses = asyncio.run(router.dispatch_many(requests, container))

    assert [resp.result["text"] for resp in responses] == [
        "You said: First",
        "You said: Second",
    ]


def test_build_default_services_provisions_router():
    """Default service container includes a prewired intent router."""
    services = build_default_services()

    assert services.intent_router is not None
    registered = services.intent_router.handlers()
    assert registered == {}
