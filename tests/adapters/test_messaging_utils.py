"""Tests for messaging adapter utilities."""

# pylint: disable=missing-function-docstring,missing-class-docstring,too-few-public-methods

from __future__ import annotations

from types import SimpleNamespace

from bt_servant_engine.adapters import messaging


def test_record_stt_usage_tracks_audio_tokens(monkeypatch) -> None:
    captured: list[tuple] = []

    def fake_add_tokens(*args, **kwargs):
        captured.append((args, kwargs))

    monkeypatch.setattr(messaging, "add_tokens", fake_add_tokens)

    usage = SimpleNamespace(
        input_tokens=10,
        output_tokens=5,
        total_tokens=15,
        audio_tokens=3,
        output_audio_tokens=2,
    )

    messaging._record_stt_usage(usage)  # pylint: disable=protected-access
    assert captured
    args, kwargs = captured[0]
    assert args[:3] == (10, 5, 15)
    assert kwargs["audio_input_tokens"] == 3
    assert kwargs["audio_output_tokens"] == 2


def test_record_stt_usage_ignores_bad_usage() -> None:
    class Explode:
        def __getattr__(self, name: str):  # noqa: D401
            raise AttributeError(name)

    # Should not raise despite missing fields
    messaging._record_stt_usage(Explode())  # pylint: disable=protected-access
