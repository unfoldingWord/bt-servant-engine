"""Unit tests for webhooks router helpers to maintain coverage."""
# pylint: disable=missing-function-docstring,too-few-public-methods,unused-argument

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import time
from contextlib import asynccontextmanager
from typing import Any

import pytest
from fastapi.testclient import TestClient

import bt_servant as api
from bt_servant_engine.apps.api.routes import webhooks
from bt_servant_engine.apps.api.state import set_brain
from bt_servant_engine.core.config import config as app_config
from bt_servant_engine.core.models import UserMessage


@asynccontextmanager
async def _noop_time_block(_: str):
    yield


def test_process_message_text_flow(monkeypatch) -> None:
    """Ensure text message processing sends responses and updates history."""
    monkeypatch.setattr(app_config, "IN_META_SANDBOX_MODE", False, raising=True)
    monkeypatch.setattr(app_config, "MESSAGE_AGE_CUTOFF_IN_SECONDS", 60, raising=True)

    message_data = {
        "from": "15555555555",
        "id": "wamid.TEXT",
        "timestamp": str(int(time.time())),
        "type": "text",
        "text": {"body": "hello"},
    }
    user_message = UserMessage.from_data(message_data)

    sent_text: list[str] = []

    async def _fake_send_text_message(user_id: str, text: str) -> None:
        sent_text.append(f"{user_id}:{text}")

    async def _fake_send_voice_message(*_, **__) -> None:
        return None

    async def _fake_sleep(*_, **__) -> None:
        return None

    async def _fake_typing_indicator(*_, **__) -> None:
        return None

    async def _fake_transcribe(*_, **__) -> str:
        return "ignored"

    monkeypatch.setattr(webhooks, "send_text_message", _fake_send_text_message)
    monkeypatch.setattr(webhooks, "send_voice_message", _fake_send_voice_message)
    monkeypatch.setattr(webhooks.asyncio, "sleep", _fake_sleep)
    monkeypatch.setattr(webhooks, "send_typing_indicator_message", _fake_typing_indicator)
    monkeypatch.setattr(webhooks, "transcribe_voice_message", _fake_transcribe)
    monkeypatch.setattr(webhooks, "get_user_chat_history", lambda **_: [])
    monkeypatch.setattr(webhooks, "update_user_chat_history", lambda **_: None)
    monkeypatch.setattr(webhooks, "get_user_response_language", lambda **_: "en")
    monkeypatch.setattr(webhooks, "get_user_agentic_strength", lambda **_: None)
    monkeypatch.setattr(webhooks, "time_block", _noop_time_block)
    monkeypatch.setattr(webhooks, "log_final_report", lambda *_, **__: None)

    class _FakeBrain:
        def invoke(self, payload: dict[str, Any]) -> dict[str, Any]:
            return {
                "translated_responses": ["response1", "response2"],
                "send_voice_message": False,
            }

    set_brain(_FakeBrain())

    asyncio.run(webhooks.process_message(user_message))

    assert len(sent_text) == 2
    set_brain(None)


def test_process_message_audio_flow(monkeypatch) -> None:
    """Audio messages trigger transcription and optional voice reply."""
    monkeypatch.setattr(app_config, "IN_META_SANDBOX_MODE", False, raising=True)
    monkeypatch.setattr(app_config, "MESSAGE_AGE_CUTOFF_IN_SECONDS", 60, raising=True)

    message_data = {
        "from": "15555555555",
        "id": "wamid.AUDIO",
        "timestamp": str(int(time.time())),
        "type": "audio",
        "audio": {"id": "media123"},
    }
    user_message = UserMessage.from_data(message_data)

    sent_text: list[str] = []
    sent_voice: list[str] = []

    async def _fake_send_text_message(user_id: str, text: str) -> None:
        sent_text.append(f"{user_id}:{text}")

    async def _fake_send_voice_message(user_id: str, text: str) -> None:
        sent_voice.append(f"{user_id}:{text}")

    async def _fake_sleep(*_, **__):
        return None

    async def _fake_typing_indicator_message(*_, **__):
        return None

    async def _fake_transcribe_voice_message(*_, **__) -> str:
        return "transcribed"

    monkeypatch.setattr(webhooks, "send_text_message", _fake_send_text_message)
    monkeypatch.setattr(webhooks, "send_voice_message", _fake_send_voice_message)
    monkeypatch.setattr(webhooks.asyncio, "sleep", _fake_sleep)
    monkeypatch.setattr(webhooks, "send_typing_indicator_message", _fake_typing_indicator_message)
    monkeypatch.setattr(webhooks, "transcribe_voice_message", _fake_transcribe_voice_message)
    monkeypatch.setattr(webhooks, "get_user_chat_history", lambda **_: [])
    monkeypatch.setattr(webhooks, "update_user_chat_history", lambda **_: None)
    monkeypatch.setattr(webhooks, "get_user_response_language", lambda **_: "en")
    monkeypatch.setattr(webhooks, "get_user_agentic_strength", lambda **_: None)
    monkeypatch.setattr(webhooks, "time_block", _noop_time_block)
    monkeypatch.setattr(webhooks, "log_final_report", lambda *_, **__: None)

    class _FakeBrain:
        def invoke(self, payload: dict[str, Any]) -> dict[str, Any]:
            return {
                "translated_responses": ["response"],
                "send_voice_message": True,
                "voice_message_text": "spoken reply",
            }

    set_brain(_FakeBrain())

    asyncio.run(webhooks.process_message(user_message))

    assert sent_text == ["15555555555:response"]
    assert sent_voice == ["15555555555:spoken reply"]
    set_brain(None)


@pytest.mark.parametrize(
    "header, secret, payload",
    [
        ("sha256", "appsecret", b"{}"),
        ("sha1", "another", b"payload"),
    ],
)
def test_verify_facebook_signature(header: str, secret: str, payload: bytes) -> None:
    """Verify signature helper supports both SHA-256 and SHA-1 headers."""
    if header == "sha256":
        expected = "sha256=" + hmac.new(secret.encode("utf-8"), payload, hashlib.sha256).hexdigest()
        assert webhooks.verify_facebook_signature(secret, payload, expected, None)
        assert not webhooks.verify_facebook_signature(secret, payload, expected + "x", None)
    else:
        expected = "sha1=" + hmac.new(secret.encode("utf-8"), payload, hashlib.sha1).hexdigest()
        assert webhooks.verify_facebook_signature(secret, payload, None, expected)
        assert not webhooks.verify_facebook_signature(secret, payload, None, expected + "x")


def test_verify_webhook_success(monkeypatch):
    monkeypatch.setattr(app_config, "META_VERIFY_TOKEN", "verify", raising=True)
    client = TestClient(api.app)
    resp = client.get(
        "/meta-whatsapp",
        params={
            "hub.mode": "subscribe",
            "hub.verify_token": "verify",
            "hub.challenge": "ok",
        },
    )
    assert resp.status_code == 200
    assert resp.text == "ok"


def test_handle_meta_webhook_invalid_signature(monkeypatch):
    monkeypatch.setattr(app_config, "META_APP_SECRET", "secret", raising=True)
    monkeypatch.setattr(app_config, "FACEBOOK_USER_AGENT", "facebookexternalua", raising=True)
    client = TestClient(api.app)
    resp = client.post(
        "/meta-whatsapp",
        headers={
            "X-Hub-Signature-256": "sha256=bad",
            "User-Agent": app_config.FACEBOOK_USER_AGENT,
        },
        json={},
    )
    assert resp.status_code == 401


def test_handle_meta_webhook_processes_message(monkeypatch):
    monkeypatch.setattr(app_config, "META_APP_SECRET", "secret", raising=True)
    monkeypatch.setattr(app_config, "FACEBOOK_USER_AGENT", "facebookexternalua", raising=True)
    monkeypatch.setattr(app_config, "IN_META_SANDBOX_MODE", False, raising=True)
    monkeypatch.setenv("RUN_OPENAI_API_TESTS", "1")

    calls: list[str] = []

    async def _fake_process_message(user_message: UserMessage) -> None:
        calls.append(user_message.user_id)

    monkeypatch.setattr(webhooks, "process_message", _fake_process_message)

    payload = {
        "entry": [
            {
                "changes": [
                    {
                        "value": {
                            "messages": [
                                {
                                    "from": "15555555555",
                                    "id": "wamid.TEST",
                                    "timestamp": str(int(time.time())),
                                    "type": "text",
                                    "text": {"body": "hi"},
                                }
                            ]
                        }
                    }
                ]
            }
        ]
    }
    body = json.dumps(payload).encode("utf-8")
    signature = "sha256=" + hmac.new(b"secret", body, hashlib.sha256).hexdigest()

    client = TestClient(api.app)
    resp = client.post(
        "/meta-whatsapp",
        headers={
            "X-Hub-Signature-256": signature,
            "User-Agent": app_config.FACEBOOK_USER_AGENT,
        },
        content=body,
    )
    assert resp.status_code == 200
    assert calls == ["15555555555"]
