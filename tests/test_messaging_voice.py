"""Tests for voice message creation: ensures filename format and mocks TTS.

This test exercises ``messaging._create_voice_message`` without making network calls
by monkeypatching the OpenAI client streaming API used in the function.
"""
# pylint: disable=too-few-public-methods,protected-access
from __future__ import annotations

import os
import re

import messaging


class _FakeStreamingCtx:
    """Context manager that writes test bytes to a target path."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def stream_to_file(self, path: str) -> None:
        """Write some bytes to the destination file path."""
        with open(path, "wb") as fh:
            fh.write(b"test-audio")


class _FakeWithStreamingResponse:
    """Shim for the OpenAI client streaming response factory."""

    def create(self, **_: object):
        """Return a fake streaming context; accepts arbitrary kwargs."""
        return _FakeStreamingCtx()


class _FakeSpeech:
    """Container exposing with_streaming_response like the real client."""

    def __init__(self) -> None:
        self.with_streaming_response = _FakeWithStreamingResponse()


class _FakeAudio:
    """OpenAI audio stub exposing a ``speech`` attribute."""

    def __init__(self) -> None:
        self.speech = _FakeSpeech()


class _FakeOpenAI:
    """Root client stub exposing an ``audio`` attribute."""

    def __init__(self) -> None:
        self.audio = _FakeAudio()


def test_create_voice_message_makes_file_and_returns_path(monkeypatch) -> None:
    """Create a voice message via mocked TTS and validate filename + contents."""
    # Arrange: stub the OpenAI client used by messaging
    monkeypatch.setattr(messaging, "open_ai_client", _FakeOpenAI(), raising=True)

    # Act
    path = messaging._create_voice_message("user123", "hello there")

    try:
        # Assert: file is created and non-empty
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0

        # Assert: filename contains expected timestamp format YYYYMMDDTHHMMSS
        base = os.path.basename(path)
        assert base.startswith("response_user123_") and base.endswith(".mp3")
        ts = base.removeprefix("response_user123_").removesuffix(".mp3")
        assert re.fullmatch(r"\d{8}T\d{6}", ts), ts
    finally:
        # Cleanup the temp file we created
        try:
            os.remove(path)
        except OSError:
            pass
