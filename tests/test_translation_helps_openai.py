"""OpenAI-backed tests for translation helps intent and flow.

These tests exercise intent classification for get-translation-helps and an API
flow using the Meta webhook with a small selection to keep token usage modest.
"""
# pylint: disable=missing-function-docstring,line-too-long,duplicate-code,unused-argument,too-many-locals
from __future__ import annotations

import os
import hmac
import hashlib
import json
import time
import re
from typing import Any, cast

from dotenv import load_dotenv
import pytest
from tinydb import TinyDB
from fastapi.testclient import TestClient

import brain
from brain import determine_intents, IntentType
import bt_servant as api
from db import user as user_db
from config import config as app_config


def _has_real_openai() -> bool:
    key = str(app_config.OPENAI_API_KEY)
    return bool(key and key != "test" and key.startswith("sk-"))


load_dotenv(override=True)


@pytest.mark.openai
@pytest.mark.skipif(not _has_real_openai(), reason="OPENAI_API_KEY not set for live OpenAI tests")
@pytest.mark.parametrize(
    "query",
    [
        "Help me translate Titus 1:1-5",
        "translation challenges for Exo 1",
        "what to consider when translating Ruth",
    ],
)
def test_intents_detect_translation_helps(query: str) -> None:
    state: dict[str, Any] = {"transformed_query": query}
    out = determine_intents(cast(Any, state))
    intents = set(out["user_intents"])  # list[IntentType]
    assert IntentType.GET_TRANSLATION_HELPS in intents


# API-level flow test (opt-in like the existing Meta test)
_RUN_OPENAI_API = os.environ.get("RUN_OPENAI_API_TESTS", "")


def _make_signature(app_secret: str, body: bytes) -> str:
    return "sha256=" + hmac.new(app_secret.encode("utf-8"), body, hashlib.sha256).hexdigest()


@pytest.mark.openai
@pytest.mark.skipif(not _has_real_openai() or _RUN_OPENAI_API != "1",
                   reason="OPENAI_API_KEY missing or RUN_OPENAI_API_TESTS!=1")
@pytest.mark.parametrize("is_first", [True, False])
def test_meta_whatsapp_translation_helps_flow_with_openai(monkeypatch, tmp_path, is_first: bool, request):
    # Ensure sandbox guard does not block the test sender
    monkeypatch.setattr(api.config, "IN_META_SANDBOX_MODE", False, raising=True)
    # Use an isolated TinyDB for user state so we can control first_interaction
    tmp_db_path = tmp_path / "db.json"
    test_db = TinyDB(str(tmp_db_path))
    request.addfinalizer(test_db.close)
    monkeypatch.setattr(user_db, "get_user_db", lambda: test_db)
    user_id = "15555555555"
    user_db.set_first_interaction(user_id, is_first)

    sent: list[str] = []

    async def _fake_send_text_message(user_id: str, text: str) -> None:  # user_id unused in test
        sent.append(text)

    async def _fake_send_voice_message(user_id: str, text: str) -> None:  # user_id unused in test
        sent.append("[voice] " + text)

    async def _fake_typing_indicator_message(message_id: str) -> None:  # message_id unused in test
        return None

    monkeypatch.setattr(api, "send_text_message", _fake_send_text_message)
    monkeypatch.setattr(api, "send_voice_message", _fake_send_voice_message)
    monkeypatch.setattr(api, "send_typing_indicator_message", _fake_typing_indicator_message)

    invoked: list[bool] = []
    orig_handler = brain.handle_get_translation_helps

    def _wrapped(state):  # type: ignore[no-redef]
        invoked.append(True)
        return orig_handler(state)

    monkeypatch.setattr(brain, "handle_get_translation_helps", _wrapped)
    api.brain = None

    def _meta_text_payload(text: str) -> dict:
        ts = str(int(time.time()))
        return {
            "entry": [
                {
                    "changes": [
                        {
                            "value": {
                                "messages": [
                                    {
                                        "from": "15555555555",
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

    with TestClient(api.app) as client:
        # Choose a tiny book to minimize tokens (e.g., 3 John)
        body_obj = _meta_text_payload("Help me translate 3 John")
        body = json.dumps(body_obj).encode("utf-8")
        app_secret = os.environ.get("META_APP_SECRET", "test")
        sig = _make_signature(app_secret, body)
        ua = os.environ.get("FACEBOOK_USER_AGENT", "test")
        resp = client.post(
            "/meta-whatsapp",
            content=body,
            headers={
                "Content-Type": "application/json",
                "X-Hub-Signature-256": sig,
                "User-Agent": ua,
            },
        )
        assert resp.status_code == 200
        deadline = time.time() + 60
        while time.time() < deadline and not sent:
            time.sleep(0.25)

    assert sent, "No outbound messages captured from translation-helps flow"
    assert invoked, "Translation helps handler was not invoked"
    combined = "\n".join(sent)
    text = re.sub(r"[^a-z0-9]+", " ", combined.lower())
    assert ("help" in text or "translate" in text) and ("3 john" in text or "3" in text)
