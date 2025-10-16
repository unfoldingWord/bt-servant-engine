"""Tests for agentic strength preference handling."""

from __future__ import annotations

from typing import Any, cast

import pytest

from bt_servant_engine.core.agentic import AgenticStrengthChoice, AgenticStrengthSetting
from bt_servant_engine.services import brain_nodes, runtime


class _StubResponse:  # pylint: disable=too-few-public-methods
    """Simple container matching the subset of Response fields we use."""

    def __init__(self, parsed: AgenticStrengthSetting) -> None:
        self.output_parsed = parsed
        self.usage = None


def test_set_agentic_strength_persists_choice(monkeypatch: pytest.MonkeyPatch) -> None:
    """When the user asks for a valid level, it is stored and echoed."""

    parsed = AgenticStrengthSetting(strength=AgenticStrengthChoice.LOW)
    monkeypatch.setattr(
        brain_nodes.open_ai_client.responses,
        "parse",
        lambda **_kwargs: _StubResponse(parsed),
    )

    captured: dict[str, Any] = {}

    def _fake_set(user_id: str, strength: str) -> None:  # noqa: D401
        captured["user_id"] = user_id
        captured["strength"] = strength

    services = runtime.get_services()
    assert services.user_state is not None
    user_state = services.user_state
    monkeypatch.setattr(user_state, "set_agentic_strength", _fake_set)

    state: dict[str, Any] = {
        "user_id": "user-123",
        "user_query": "set my agentic strength to low",
        "user_chat_history": [],
    }

    out = brain_nodes.set_agentic_strength(cast(Any, state))

    assert captured == {"user_id": "user-123", "strength": "low"}
    assert out.get("agentic_strength") == "low"
    assert out.get("user_agentic_strength") == "low"
    response_text = out["responses"][0]["response"]
    assert "agentic strength set to" in response_text.lower()
    assert "low" in response_text.lower()


def test_set_agentic_strength_handles_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    """If the LLM can't find a valid level, prompt the user again."""

    parsed = AgenticStrengthSetting(strength=AgenticStrengthChoice.UNKNOWN)
    monkeypatch.setattr(
        brain_nodes.open_ai_client.responses,
        "parse",
        lambda **_kwargs: _StubResponse(parsed),
    )

    called = False

    def _fail_set(*_args: Any, **_kwargs: Any) -> None:  # noqa: ANN401
        nonlocal called
        called = True

    services = runtime.get_services()
    assert services.user_state is not None
    user_state = services.user_state
    monkeypatch.setattr(user_state, "set_agentic_strength", _fail_set)

    state: dict[str, Any] = {
        "user_id": "user-456",
        "user_query": "make it stronger",
        "user_chat_history": [],
    }

    out = brain_nodes.set_agentic_strength(cast(Any, state))

    assert not called, "should not persist an unknown preference"
    assert "normal, low, or very low" in out["responses"][0]["response"]
    assert "agentic_strength" not in out
