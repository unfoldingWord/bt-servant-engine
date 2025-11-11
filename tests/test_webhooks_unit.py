"""Unit tests for webhooks router helpers to maintain coverage."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import time
from contextlib import asynccontextmanager
from types import SimpleNamespace

from http import HTTPStatus

import pytest
from fastapi.testclient import TestClient

from bt_servant_engine.apps.api.app import create_app
from bt_servant_engine.apps.api.routes import webhooks
from bt_servant_engine.apps.api.state import set_brain
from bt_servant_engine.core.config import config as app_config
from bt_servant_engine.core.models import UserMessage
from bt_servant_engine.services import runtime


@asynccontextmanager
async def _noop_time_block(_: str):
    yield


async def _noop_sleep(*args, **kwargs) -> None:
    del args, kwargs


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

    services = runtime.get_services()
    messaging = services.messaging
    assert messaging is not None
    messaging.sent_text.clear()  # type: ignore[attr-defined]
    messaging.sent_voice.clear()  # type: ignore[attr-defined]
    messaging.typing.clear()  # type: ignore[attr-defined]

    monkeypatch.setattr(webhooks.asyncio, "sleep", _noop_sleep)
    monkeypatch.setattr(webhooks, "time_block", _noop_time_block)
    monkeypatch.setattr(webhooks, "log_final_report", lambda *_, **__: None)

    brain = SimpleNamespace(
        invoke=lambda payload: {
            "translated_responses": ["response1", "response2"],
            "send_voice_message": False,
        }
    )
    set_brain(brain)

    asyncio.run(
        webhooks.process_message(
            user_message,
            services,
            correlation_id="test-cid",
            client_ip="203.0.113.10",
        )
    )

    assert messaging.sent_text == [  # type: ignore[attr-defined]
        ("15555555555", "(1/2) response1"),
        ("15555555555", "(2/2) response2"),
    ]
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

    services = runtime.get_services()
    messaging = services.messaging
    assert messaging is not None
    messaging.sent_text.clear()  # type: ignore[attr-defined]
    messaging.sent_voice.clear()  # type: ignore[attr-defined]
    messaging.typing.clear()  # type: ignore[attr-defined]

    async def _fake_transcribe_voice_message(*args, **kwargs) -> str:
        del args, kwargs
        return "transcribed"

    monkeypatch.setattr(webhooks.asyncio, "sleep", _noop_sleep)
    monkeypatch.setattr(messaging, "transcribe_voice_message", _fake_transcribe_voice_message)
    monkeypatch.setattr(webhooks, "time_block", _noop_time_block)
    monkeypatch.setattr(webhooks, "log_final_report", lambda *_, **__: None)

    brain = SimpleNamespace(
        invoke=lambda payload: {
            "translated_responses": ["response"],
            "send_voice_message": True,
            "voice_message_text": "spoken reply",
        }
    )
    set_brain(brain)

    asyncio.run(
        webhooks.process_message(
            user_message,
            services,
            correlation_id="test-cid",
            client_ip="203.0.113.10",
        )
    )

    assert messaging.sent_text == [  # type: ignore[attr-defined]
        (
            "15555555555",
            "_I'm transcribing your voice message. Give me a moment._",
        ),
        (
            "15555555555",
            "_I'm packaging up a voice message response._",
        ),
        ("15555555555", "response"),
    ]
    assert messaging.sent_voice == [  # type: ignore[attr-defined]
        ("15555555555", "spoken reply")
    ]
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
    """Verification endpoint echoes the hub challenge on success."""
    monkeypatch.setattr(app_config, "META_VERIFY_TOKEN", "verify", raising=True)
    client = TestClient(create_app(services=runtime.get_services()))
    resp = client.get(
        "/meta-whatsapp",
        params={
            "hub.mode": "subscribe",
            "hub.verify_token": "verify",
            "hub.challenge": "ok",
        },
    )
    assert resp.status_code == HTTPStatus.OK
    assert resp.text == "ok"


def test_handle_meta_webhook_invalid_signature(monkeypatch):
    """Webhook rejects requests with an invalid signature."""
    monkeypatch.setattr(app_config, "META_APP_SECRET", "secret", raising=True)
    monkeypatch.setattr(app_config, "FACEBOOK_USER_AGENT", "facebookexternalua", raising=True)
    client = TestClient(create_app(services=runtime.get_services()))
    resp = client.post(
        "/meta-whatsapp",
        headers={
            "X-Hub-Signature-256": "sha256=bad",
            "User-Agent": app_config.FACEBOOK_USER_AGENT,
        },
        json={},
    )
    assert resp.status_code == HTTPStatus.UNAUTHORIZED


def test_handle_meta_webhook_processes_message(monkeypatch):
    """Webhook accepts valid signatures and invokes process_message."""
    monkeypatch.setattr(app_config, "META_APP_SECRET", "secret", raising=True)
    monkeypatch.setattr(app_config, "FACEBOOK_USER_AGENT", "facebookexternalua", raising=True)
    monkeypatch.setattr(app_config, "IN_META_SANDBOX_MODE", False, raising=True)
    monkeypatch.setenv("RUN_OPENAI_API_TESTS", "1")

    calls: list[str] = []

    async def _fake_process_message(  # noqa: ANN001
        user_message: UserMessage,
        *,
        correlation_id: str | None,
        client_ip: str | None,
        **_,
    ) -> None:
        calls.append(user_message.user_id)
        assert correlation_id == "test-cid"
        assert client_ip == "testclient"

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

    client = TestClient(create_app(services=runtime.get_services()))
    resp = client.post(
        "/meta-whatsapp",
        headers={
            "X-Hub-Signature-256": signature,
            "User-Agent": app_config.FACEBOOK_USER_AGENT,
            "X-Request-ID": "test-cid",
        },
        content=body,
    )
    assert resp.status_code == HTTPStatus.OK
    assert calls == ["15555555555"]
