"""Tests for core data models."""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from bt_servant_engine.core import models


def test_user_message_from_data_text_and_audio() -> None:
    """Factory builds message objects for text and audio payloads."""
    now = int(time.time())
    text_payload = {
        "from": "user-1",
        "id": "msg-1",
        "type": "text",
        "timestamp": str(now),
        "text": {"body": "Hello"},
    }
    msg = models.UserMessage.from_data(text_payload)
    assert msg.text == "Hello"
    assert msg.media_id == ""
    assert msg.is_supported_type() is True

    audio_payload = {
        "from": "user-2",
        "id": "msg-2",
        "type": "audio",
        "timestamp": str(now),
        "audio": {"id": "media-123"},
    }
    audio_msg = models.UserMessage.from_data(audio_payload)
    assert audio_msg.media_id == "media-123"


def test_user_message_validations() -> None:
    """Factory rejects malformed payloads across supported message types."""
    now = int(time.time())
    with pytest.raises(ValueError):
        models.UserMessage.from_data({"id": "x", "type": "text", "timestamp": str(now)})

    with pytest.raises(ValueError):
        models.UserMessage.from_data({"from": "u", "type": "text", "timestamp": str(now)})

    with pytest.raises(ValueError):
        models.UserMessage.from_data({"from": "u", "id": "x", "timestamp": str(now), "type": None})

    with pytest.raises(ValueError):
        models.UserMessage.from_data({"from": "u", "id": "x", "type": "text", "timestamp": ""})

    with pytest.raises(ValueError):
        models.UserMessage.from_data(
            {"from": "u", "id": "x", "type": "text", "timestamp": str(now), "text": {}},
        )

    with pytest.raises(ValueError):
        models.UserMessage.from_data(
            {"from": "u", "id": "x", "type": "audio", "timestamp": str(now), "audio": {}},
        )

    with pytest.raises(ValueError):
        models.UserMessage.from_data(
            {"from": "u", "id": "x", "type": "text", "timestamp": str(now), "text": "oops"},
        )

    with pytest.raises(ValueError):
        models.UserMessage.from_data(
            {"from": "u", "id": "x", "type": "audio", "timestamp": str(now), "audio": "oops"},
        )


def test_user_message_age_and_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    """Age and sandbox flags flag outdated or unauthorized messages."""
    fake_config = SimpleNamespace(
        MESSAGE_AGE_CUTOFF_IN_SECONDS=5,
        IN_META_SANDBOX_MODE=True,
        META_SANDBOX_PHONE_NUMBER="allowed",
    )
    monkeypatch.setattr("bt_servant_engine.core.config.config", fake_config)
    monkeypatch.setattr("bt_servant_engine.core.models.config", fake_config)

    now = int(time.time())
    stale_timestamp = now - fake_config.MESSAGE_AGE_CUTOFF_IN_SECONDS - 1
    msg = models.UserMessage("id", "user", "text", stale_timestamp, text="hi")
    assert msg.age() >= 0

    assert msg.too_old() is True

    msg_sandbox = models.UserMessage("id2", "blocked", "text", now, text="hi")
    assert msg_sandbox.is_unauthorized_sender() is True
