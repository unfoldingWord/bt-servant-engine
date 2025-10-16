"""OpenAI-backed tests for translation helps intent and flow.

These tests exercise intent classification for get-translation-helps and an API
flow using the Meta webhook with a small selection to keep token usage modest.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import time
from typing import Any, cast

from http import HTTPStatus

import pytest
from dotenv import load_dotenv
from fastapi.testclient import TestClient
from tinydb import TinyDB

from bt_servant_engine.apps.api.app import create_app
from bt_servant_engine.apps.api.routes import webhooks
from bt_servant_engine.apps.api.state import set_brain
from bt_servant_engine.bootstrap import build_default_service_container
from bt_servant_engine.core.config import config as app_config
from bt_servant_engine.core.intents import IntentType
from bt_servant_engine.services import brain_nodes, runtime
from bt_servant_engine.services.brain_nodes import determine_intents
import bt_servant_engine.adapters.user_state as user_db


def _has_real_openai() -> bool:
    key = str(app_config.OPENAI_API_KEY)
    return bool(key and key != "test" and key.startswith("sk-"))


load_dotenv(override=True)

_TEST_USER_ID = "15555555555"


@pytest.mark.openai
@pytest.mark.skipif(not _has_real_openai(), reason="OPENAI_API_KEY not set for live OpenAI tests")
@pytest.mark.parametrize(
    "query,acceptable_intents",
    [
        # Ambiguous: could be general translation help or structured translation helps
        (
            "Help me translate Titus 1:1-5",
            {IntentType.GET_TRANSLATION_HELPS, IntentType.GET_BIBLE_TRANSLATION_ASSISTANCE},
        ),
        (
            "translation challenges for Exo 1",
            {IntentType.GET_TRANSLATION_HELPS, IntentType.GET_BIBLE_TRANSLATION_ASSISTANCE},
        ),
        (
            "what to consider when translating Ruth",
            {IntentType.GET_TRANSLATION_HELPS, IntentType.GET_BIBLE_TRANSLATION_ASSISTANCE},
        ),
    ],
)
def test_intents_detect_translation_helps(query: str, acceptable_intents: set[IntentType]) -> None:
    """Structured translation-helps queries should produce related intents."""
    state: dict[str, Any] = {"transformed_query": query}
    out = determine_intents(cast(Any, state))
    intents = set(out["user_intents"])  # list[IntentType]
    assert any(
        intent in intents for intent in acceptable_intents
    ), f"Expected one of {acceptable_intents}, got {intents}"


# API-level flow test (opt-in like the existing Meta test)
_RUN_OPENAI_API = os.environ.get("RUN_OPENAI_API_TESTS", "")


def _make_signature(app_secret: str, body: bytes) -> str:
    return "sha256=" + hmac.new(app_secret.encode("utf-8"), body, hashlib.sha256).hexdigest()


def _initialize_user_state(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any, is_first: bool, request: Any
) -> None:
    """Seed user state storage for translation helps tests."""

    tmp_db_path = tmp_path / "db.json"
    test_db = TinyDB(str(tmp_db_path))
    request.addfinalizer(test_db.close)
    monkeypatch.setattr(user_db, "get_user_db", lambda: test_db)
    user_db.set_first_interaction(_TEST_USER_ID, is_first)


def _patch_messaging(monkeypatch: pytest.MonkeyPatch, services: Any, sent: list[str]) -> None:
    """Patch messaging send hooks to capture outputs."""

    messaging = services.messaging
    if messaging is None:
        raise RuntimeError("Messaging service is not configured.")

    async def send_text_message(_user_id: str, text: str) -> None:
        sent.append(text)

    async def send_voice_message(_user_id: str, text: str) -> None:
        sent.append(f"[voice] {text}")

    async def send_typing_indicator(_message_id: str) -> None:
        return None

    monkeypatch.setattr(messaging, "send_text_message", send_text_message)
    monkeypatch.setattr(messaging, "send_voice_message", send_voice_message)
    monkeypatch.setattr(messaging, "send_typing_indicator", send_typing_indicator)


def _wrap_translation_handler(monkeypatch: pytest.MonkeyPatch) -> list[bool]:
    """Wrap translation helps handler to track invocation."""

    invoked: list[bool] = []
    orig_handler = brain_nodes.handle_get_translation_helps

    def _wrapped(state: Any) -> Any:
        invoked.append(True)
        return orig_handler(state)

    monkeypatch.setattr(brain_nodes, "handle_get_translation_helps", _wrapped)
    set_brain(None)
    return invoked


def _build_meta_payload(text: str) -> dict:
    """Construct a Meta webhook payload for the test user."""

    ts = str(int(time.time()))
    return {
        "entry": [
            {
                "changes": [
                    {
                        "value": {
                            "messages": [
                                {
                                    "from": _TEST_USER_ID,
                                    "id": "wamid.TEST",
                                    "timestamp": ts,
                                    "type": "text",
                                    "text": {"body": text},
                                }
                            ]
                        }
                    }
                ]
            }
        ]
    }


def _post_meta_request(client: TestClient, payload: dict) -> Any:
    """Send a Meta webhook request with appropriate headers."""

    body = json.dumps(payload).encode("utf-8")
    app_secret = os.environ.get("META_APP_SECRET", "test")
    signature = _make_signature(app_secret, body)
    user_agent = os.environ.get("FACEBOOK_USER_AGENT", "test")
    return client.post(
        "/meta-whatsapp",
        content=body,
        headers={
            "Content-Type": "application/json",
            "X-Hub-Signature-256": signature,
            "User-Agent": user_agent,
        },
    )


def _wait_for_messages(sent: list[str], timeout: float = 60.0) -> None:
    """Spin until captured messages are available or the timeout elapses."""

    deadline = time.time() + timeout
    while time.time() < deadline and not sent:
        time.sleep(0.25)


@pytest.mark.openai
@pytest.mark.skipif(
    not _has_real_openai() or _RUN_OPENAI_API != "1",
    reason="OPENAI_API_KEY missing or RUN_OPENAI_API_TESTS!=1",
)
@pytest.mark.xfail(
    reason="TODO: Needs update for sequential intent queue system (PR #128). "
    "Query 'Help me translate 3 John' may trigger multiple intents now."
)
@pytest.mark.parametrize("is_first", [True, False])
def test_meta_whatsapp_translation_helps_flow_with_openai(
    monkeypatch, tmp_path, is_first: bool, request
):
    """End-to-end Meta WhatsApp flow sends translation helps via OpenAI."""
    monkeypatch.setattr(app_config, "IN_META_SANDBOX_MODE", False, raising=True)
    _initialize_user_state(monkeypatch, tmp_path, is_first, request)

    sent: list[str] = []
    invoked = _wrap_translation_handler(monkeypatch)

    with TestClient(create_app(build_default_service_container())) as client:
        services = runtime.get_services()
        _patch_messaging(monkeypatch, services, sent)
        monkeypatch.setattr(webhooks.asyncio, "sleep", lambda *_, **__: None)
        payload = _build_meta_payload("Help me translate 3 John")
        resp = _post_meta_request(client, payload)
        assert resp.status_code == HTTPStatus.OK
        _wait_for_messages(sent)

    assert sent, "No outbound messages captured from translation-helps flow"
    assert invoked, "Translation helps handler was not invoked"
    combined = "\n".join(sent)
    text = re.sub(r"[^a-z0-9]+", " ", combined.lower())
    assert ("help" in text or "translate" in text) and ("3 john" in text or "3" in text)
