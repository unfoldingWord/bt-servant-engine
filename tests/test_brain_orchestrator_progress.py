"""Tests for progress-aware node wrapping in the brain orchestrator."""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List

from bt_servant_engine.services.brain_orchestrator import wrap_node_with_progress


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

    messages: List[str] = []

    async def messenger(message: str) -> None:
        messages.append(message)

    state = _make_state(messenger)

    wrapped = wrap_node_with_progress(lambda s: s, "test_node", progress_message="Working...")

    wrapped(state)

    assert messages == ["Working..."]
    assert state["last_progress_time"] > 0


def test_wrap_node_with_progress_schedules_on_running_loop() -> None:
    """Progress messages use the current loop when one is already running."""

    messages: List[str] = []

    async def messenger(message: str) -> None:
        messages.append(message)

    state = _make_state(messenger)

    wrapped = wrap_node_with_progress(lambda s: s, "test_node", progress_message="Working...")

    async def run() -> None:
        wrapped(state)
        await asyncio.sleep(0)

    asyncio.run(run())

    assert messages == ["Working..."]
    assert state["last_progress_time"] > 0
