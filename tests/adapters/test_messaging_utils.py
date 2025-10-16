"""Tests for messaging adapter utilities."""

# pylint: disable=missing-function-docstring,missing-class-docstring,too-few-public-methods

from __future__ import annotations

from types import SimpleNamespace

from bt_servant_engine.adapters import messaging

INPUT_TOKENS = 10
OUTPUT_TOKENS = 5
TOTAL_TOKENS = 15
AUDIO_INPUT_TOKENS = 3
AUDIO_OUTPUT_TOKENS = 2


def test_record_stt_usage_tracks_audio_tokens(monkeypatch) -> None:
    captured: list = []

    def fake_add_tokens(increments):
        captured.append(increments)

    monkeypatch.setattr(messaging, "add_tokens", fake_add_tokens)

    usage = SimpleNamespace(
        input_tokens=INPUT_TOKENS,
        output_tokens=OUTPUT_TOKENS,
        total_tokens=TOTAL_TOKENS,
        audio_tokens=AUDIO_INPUT_TOKENS,
        output_audio_tokens=AUDIO_OUTPUT_TOKENS,
    )

    messaging._record_stt_usage(usage)  # pylint: disable=protected-access
    assert captured
    increments = captured[0]
    assert getattr(increments, "input_tokens", None) == INPUT_TOKENS
    assert getattr(increments, "output_tokens", None) == OUTPUT_TOKENS
    assert getattr(increments, "total_tokens", None) == TOTAL_TOKENS
    assert getattr(increments, "audio_input_tokens", None) == AUDIO_INPUT_TOKENS
    assert getattr(increments, "audio_output_tokens", None) == AUDIO_OUTPUT_TOKENS


def test_record_stt_usage_ignores_bad_usage() -> None:
    class Explode:
        def __getattr__(self, name: str):  # noqa: D401
            raise AttributeError(name)

    # Should not raise despite missing fields
    messaging._record_stt_usage(Explode())  # pylint: disable=protected-access
