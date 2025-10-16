"""Tests for progress-aware node wrapping in the brain orchestrator."""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List

from bt_servant_engine.services import status_messages
from bt_servant_engine.services.brain_orchestrator import (
    build_translation_assistance_progress_message,
    wrap_node_with_progress,
)
from bt_servant_engine.services.progress_messaging import should_show_translation_progress


def _make_state(messenger):
    state: Dict[str, Any] = {
        "progress_enabled": True,
        "progress_messenger": messenger,
        "progress_throttle_seconds": 0,
        "last_progress_time": 0,
    }
    return state


def test_wrap_node_with_progress_runs_without_event_loop() -> None:
    """Progress messages run via `asyncio.run` when no loop is active."""

    messages: List[status_messages.LocalizedProgressMessage] = []

    async def messenger(message: status_messages.LocalizedProgressMessage) -> None:
        messages.append(message)

    state = _make_state(messenger)

    wrapped = wrap_node_with_progress(lambda s: s, "test_node", progress_message="Working...")

    wrapped(state)

    assert [m["text"] for m in messages] == ["Working..."]
    assert [m["emoji"] for m in messages] == ["⏳"]
    assert state["last_progress_time"] > 0


def test_should_show_translation_progress_skips_english_codes() -> None:
    """Progress messages for translation do not trigger for English targets."""

    state: Dict[str, Any] = {
        "responses": [{"response": "example"}],
        "user_response_language": "en",
    }

    assert should_show_translation_progress(state) is False

    state["user_response_language"] = "English"
    assert should_show_translation_progress(state) is False

    state["user_response_language"] = "English (US)"
    assert should_show_translation_progress(state) is False


def test_wrap_node_with_progress_schedules_on_running_loop() -> None:
    """Progress messages use the current loop when one is already running."""

    messages: List[status_messages.LocalizedProgressMessage] = []

    async def messenger(message: status_messages.LocalizedProgressMessage) -> None:
        messages.append(message)

    state = _make_state(messenger)

    wrapped = wrap_node_with_progress(lambda s: s, "test_node", progress_message="Working...")

    async def run() -> None:
        wrapped(state)
        await asyncio.sleep(0)

    asyncio.run(run())

    assert [m["text"] for m in messages] == ["Working..."]
    assert [m["emoji"] for m in messages] == ["⏳"]
    assert state["last_progress_time"] > 0


def test_translation_progress_message_includes_sources() -> None:
    """Translation assistance progress message lists unique resource origins."""

    messages: List[status_messages.LocalizedProgressMessage] = []

    async def messenger(message: status_messages.LocalizedProgressMessage) -> None:
        messages.append(message)

    state = _make_state(messenger)
    state["docs"] = [
        {"metadata": {"_merged_from": "uw_notes"}},
        {"metadata": {"_merged_from": "uw_dictionary"}},
        {"metadata": {"_merged_from": "uw_notes"}},  # duplicate omitted
    ]

    wrapped = wrap_node_with_progress(
        lambda s: s,
        "test_node",
        progress_message=build_translation_assistance_progress_message,
    )

    wrapped(state)

    expected_text = (
        "I found potentially relevant documents in the following resources: "
        "uw notes and uw dictionary. I'm pulling everything together into a helpful response "
        "for you."
    )
    assert [m["text"] for m in messages] == [expected_text]
    assert [m["emoji"] for m in messages] == ["📚"]


def test_translation_progress_message_falls_back_without_sources() -> None:
    """Translation assistance progress message reuses base text when no sources."""

    messages: List[status_messages.LocalizedProgressMessage] = []

    async def messenger(message: status_messages.LocalizedProgressMessage) -> None:
        messages.append(message)

    state = _make_state(messenger)
    state["docs"] = []

    wrapped = wrap_node_with_progress(
        lambda s: s,
        "test_node",
        progress_message=build_translation_assistance_progress_message,
    )

    wrapped(state)

    assert [m["text"] for m in messages] == [
        "I'm pulling everything together into a helpful response for you."
    ]
    assert [m["emoji"] for m in messages] == ["⏳"]
