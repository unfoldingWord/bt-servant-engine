"""Unit tests for consult_fia_resources intent handling."""

from __future__ import annotations

from typing import Any, cast

import pytest

import brain


class _FakeCollection:  # pylint: disable=too-few-public-methods
    """Simple stand-in for a Chroma collection."""

    def __init__(self, documents: list[tuple[str, float, dict[str, str]]]) -> None:
        self._documents = documents

    def query(self, *, query_texts: list[str], n_results: int) -> dict[str, Any]:  # noqa: ARG002
        """Return a deterministic subset of stored docs for the test."""
        _ = query_texts
        docs = [doc for doc, _dist, _meta in self._documents][:n_results]
        distances = [dist for _doc, dist, _meta in self._documents][:n_results]
        metadatas = [meta for _doc, _dist, meta in self._documents][:n_results]
        return {
            "documents": [docs],
            "distances": [distances],
            "metadatas": [metadatas],
        }


def test_consult_fia_resources_falls_back_to_english(monkeypatch: pytest.MonkeyPatch) -> None:
    """Falls back to the English FIA collection when localized resources are missing."""

    captured_messages: dict[str, Any] = {}

    def _fake_get_collection(name: str) -> _FakeCollection | None:
        if name == "en_fia_resources":
            docs = [
                ("English FIA doc", 0.1, {"name": "EN FIA", "source": "fia/en.doc"}),
            ]
            return _FakeCollection(docs)
        return None

    class _FakeResponse:  # pylint: disable=too-few-public-methods
        usage = None
        output_text = "synthesized fia answer"

    def _fake_create(**kwargs: Any) -> Any:  # noqa: ANN401
        captured_messages["input"] = kwargs["input"]
        return _FakeResponse()

    monkeypatch.setattr(brain, "get_chroma_collection", _fake_get_collection)
    monkeypatch.setattr(brain.open_ai_client.responses, "create", _fake_create)
    monkeypatch.setattr(brain, "FIA_REFERENCE_CONTENT", "FIA manual reference text")

    state: dict[str, Any] = {
        "transformed_query": "What does FIA step 2 look like in Mark 1?",
        "user_chat_history": [],
        "user_response_language": "es",
        "query_language": "es",
    }

    out = brain.consult_fia_resources(cast(Any, state))

    assert out["responses"][0]["intent"] == brain.IntentType.CONSULT_FIA_RESOURCES
    assert out.get("collection_used") == "en_fia_resources"

    # Ensure the FIA manual and vector doc both reached the prompt payload
    payload = captured_messages["input"][0]["content"]
    assert "FIA manual reference text" in payload
    assert "English FIA doc" in payload


def test_consult_fia_resources_handles_missing_context(monkeypatch: pytest.MonkeyPatch) -> None:
    """Returns a helpful fallback when no FIA resources are available."""

    monkeypatch.setattr(brain, "get_chroma_collection", lambda _name: None)
    monkeypatch.setattr(brain, "FIA_REFERENCE_CONTENT", "")

    state: dict[str, Any] = {
        "transformed_query": "How do I translate the Bible?",
        "user_chat_history": [],
        "user_response_language": None,
        "query_language": "en",
    }

    out = brain.consult_fia_resources(cast(Any, state))

    response_text = out["responses"][0]["response"]
    assert "couldn't find any FIA resources" in response_text
    assert brain.IntentType.CONSULT_FIA_RESOURCES == out["responses"][0]["intent"]
