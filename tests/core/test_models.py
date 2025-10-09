"""Tests for core data models."""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from bt_servant_engine.core import models

# pylint: disable=missing-function-docstring


def test_user_message_from_data_text_and_audio(monkeypatch: pytest.MonkeyPatch) -> None:
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


def test_user_message_validations(monkeypatch: pytest.MonkeyPatch) -> None:
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
    now = int(time.time())
    msg = models.UserMessage("id", "user", "text", str(now - 10), text="hi")
    assert msg.age() >= 0

    fake_config = SimpleNamespace(
        MESSAGE_AGE_CUTOFF_IN_SECONDS=5,
        IN_META_SANDBOX_MODE=True,
        META_SANDBOX_PHONE_NUMBER="allowed",
    )
    monkeypatch.setattr("bt_servant_engine.core.config.config", fake_config)

    assert msg.too_old() is True

    msg_sandbox = models.UserMessage("id2", "blocked", "text", str(now), text="hi")
    assert msg_sandbox.is_unauthorized_sender() is True
