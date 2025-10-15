"""Tests for intent processing and follow-up suppression logic."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import pytest

from bt_servant_engine.core.intents import IntentType, IntentWithContext
from bt_servant_engine.services import brain_nodes, brain_orchestrator


@pytest.fixture()
def capture_intent_queue(monkeypatch: pytest.MonkeyPatch) -> Dict[str, List[Any]]:
    """Patch intent queue helpers for deterministic tests."""

    captured: Dict[str, List[Any]] = {}

    def fake_save(user_id: str, items: List[Any]) -> None:
        captured[user_id] = items

    monkeypatch.setattr(
        "bt_servant_engine.services.intent_queue.save_intent_queue",
        fake_save,
    )
    monkeypatch.setattr(
        "bt_servant_engine.services.intent_queue.pop_next_intent",
        lambda _user_id: None,
    )
    # Default to no additional queued intents unless overridden in a test
    monkeypatch.setattr(
        "bt_servant_engine.services.intent_queue.has_queued_intents",
        lambda _user_id: False,
    )
    return captured


def test_process_intents_sets_active_query_and_suppresses_followup(  # pylint: disable=redefined-outer-name
    capture_intent_queue: Dict[str, List[Any]],
) -> None:
    """Multi-intent flows should focus the active query and defer follow-ups."""

    state: Dict[str, Any] = {
        "user_id": "user-1",
        "user_query": "Can you tell me how the FIA process works and also tell me about Barnabas from the Bible?",
        "transformed_query": "Can you tell me how the FIA process works and also tell me about Barnabas from the Bible?",
        "user_intents": [
            IntentType.CONSULT_FIA_RESOURCES,
            IntentType.GET_BIBLE_TRANSLATION_ASSISTANCE,
        ],
        "intents_with_context": [
            IntentWithContext(intent=IntentType.CONSULT_FIA_RESOURCES, parameters_json="{}"),
            IntentWithContext(
                intent=IntentType.GET_BIBLE_TRANSLATION_ASSISTANCE,
                parameters_json=json.dumps({"query": "Barnabas from the Bible"}),
            ),
        ],
        "continuation_actions": [
            "Would you like me to explain how the FIA process works?",
            "Would you like me to tell you about Barnabas from the Bible?",
        ],
    }

    nodes = brain_orchestrator.process_intents(state)

    # Highest priority intent should be FIA resources node
    assert nodes == ["consult_fia_resources_node"]
    assert state["active_intent_query"] == "Explain how the FIA process works"
    assert state["suppress_internal_followups"] is True
    assert state["has_more_queued_intents"] is True
    assert state["deferred_intent_topics"] == ["Barnabas from the Bible"]

    queued = capture_intent_queue.get("user-1")
    assert queued is not None and len(queued) == 1
    queued_item = queued[0]
    assert queued_item.intent is IntentType.GET_BIBLE_TRANSLATION_ASSISTANCE
    assert queued_item.parameters["query"] == "Barnabas from the Bible"


def test_process_intents_for_queued_intent_uses_context(monkeypatch: pytest.MonkeyPatch) -> None:
    """When processing a queued intent, prefer stored parameters and allow follow-ups when queue empties."""

    monkeypatch.setattr(
        "bt_servant_engine.services.intent_queue.pop_next_intent",
        lambda _user_id: None,
    )
    monkeypatch.setattr(
        "bt_servant_engine.services.intent_queue.has_queued_intents",
        lambda _user_id: False,
    )

    state: Dict[str, Any] = {
        "user_id": "user-2",
        "user_query": "yes please",
        "transformed_query": "yes please",
        "user_intents": [IntentType.GET_BIBLE_TRANSLATION_ASSISTANCE],
        "queued_intent_context": {"query": "Barnabas from the Bible"},
    }

    nodes = brain_orchestrator.process_intents(state)

    assert nodes == ["query_vector_db_node"]
    assert state["active_intent_query"] == "Barnabas from the Bible"
    assert state["suppress_internal_followups"] is False
    assert state["has_more_queued_intents"] is False


def test_query_open_ai_respects_followup_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure brain_nodes.query_open_ai forwards the focused query and follow-up preference."""

    captured: Dict[str, Any] = {}

    def fake_query_open_ai_impl(
        _client: Any,
        docs: List[dict[str, Any]],
        transformed_query: str,
        chat_history: List[dict[str, str]],
        model_for_agentic_strength_fn: Any,
        extract_cached_input_tokens_fn: Any,
        add_tokens_fn: Any,
        agentic_strength: str,
        boilerplate_features_message: str,
        *,
        include_followup: bool = True,
        ignored_topics: Optional[List[str]] = None,
    ) -> dict[str, Any]:
        captured["docs"] = docs
        captured["query"] = transformed_query
        captured["include_followup"] = include_followup
        captured["ignored_topics"] = ignored_topics
        return {"responses": [{"intent": IntentType.GET_BIBLE_TRANSLATION_ASSISTANCE, "response": "ok"}]}

    monkeypatch.setattr(brain_nodes, "query_open_ai_impl", fake_query_open_ai_impl)

    state: Dict[str, Any] = {
        "transformed_query": "Original question",
        "active_intent_query": "Focused query",
        "user_chat_history": [],
        "docs": [{"resource_name": "Example", "source": "Test"}],
        "agentic_strength": "normal",
        "suppress_internal_followups": True,
    }

    result = brain_nodes.query_open_ai(state)

    assert captured["query"] == "Focused query"
    assert captured["include_followup"] is False
    assert captured["ignored_topics"] == []
    assert result["responses"][0]["response"] == "ok"
