"""Unit tests for consult_fia_resources intent handling."""

from __future__ import annotations

from typing import Any, Iterable, Mapping, cast

import pytest

from bt_servant_engine.core.intents import IntentType
from bt_servant_engine.core.exceptions import CollectionNotFoundError
from bt_servant_engine.core.ports import ChromaPort
from bt_servant_engine.services import brain_nodes, runtime


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


class StubChromaPort(ChromaPort):
    """Minimal ChromaPort stub for consult FIA tests."""

    def __init__(self, mapping: Mapping[str, _FakeCollection | None] | None = None) -> None:
        self._mapping = dict(mapping or {})

    def get_collection(self, name: str) -> _FakeCollection | None:  # noqa: D401
        return self._mapping.get(name)

    def list_collections(self) -> list[str]:
        return list(self._mapping.keys())

    def create_collection(self, name: str) -> None:  # noqa: D401
        self._mapping.setdefault(name, None)

    def delete_collection(self, name: str) -> None:  # noqa: D401
        self._mapping.pop(name, None)

    def delete_document(self, name: str, document_id: str) -> None:  # noqa: D401
        del name, document_id

    def count_documents(self, name: str) -> int:
        del name
        return 0

    def get_document_text_and_metadata(
        self, name: str, document_id: str
    ) -> tuple[str, Mapping[str, Any]]:
        del name, document_id
        return "", {}

    def list_document_ids(self, name: str) -> list[str]:
        del name
        return []

    def iter_batches(
        self,
        name: str,
        *,
        batch_size: int = 1000,
        include_embeddings: bool = False,
    ) -> Iterable[dict[str, Any]]:
        del name, batch_size, include_embeddings
        return []

    def get_collections_pair(self, source: str, dest: str) -> tuple[Any, Any]:
        src = self.get_collection(source)
        dst = self.get_collection(dest)
        if src is None or dst is None:
            raise CollectionNotFoundError("Missing collection for merge")
        return src, dst

    def max_numeric_id(self, name: str) -> int:
        del name
        return 0


def test_consult_fia_resources_falls_back_to_english(monkeypatch: pytest.MonkeyPatch) -> None:
    """Falls back to the English FIA collection when localized resources are missing."""

    captured_messages: dict[str, Any] = {}

    services = runtime.get_services()
    services.chroma = StubChromaPort(
        {
            "es_fia_resources": None,
            "en_fia_resources": _FakeCollection(
                [("English FIA doc", 0.1, {"name": "EN FIA", "source": "fia/en.doc"})]
            ),
        }
    )

    class _FakeResponse:  # pylint: disable=too-few-public-methods
        usage = None
        output_text = "synthesized fia answer"

    def _fake_create(**kwargs: Any) -> Any:  # noqa: ANN401
        captured_messages["input"] = kwargs["input"]
        captured_messages["model"] = kwargs.get("model")
        return _FakeResponse()

    from bt_servant_engine.services.intents import fia_intents

    monkeypatch.setattr(brain_nodes.open_ai_client.responses, "create", _fake_create)
    monkeypatch.setattr(fia_intents, "FIA_REFERENCE_CONTENT", "FIA manual reference text")

    state: dict[str, Any] = {
        "transformed_query": "What does FIA step 2 look like in Mark 1?",
        "user_chat_history": [],
        "user_response_language": "es",
        "query_language": "es",
        "agentic_strength": "low",
    }

    out = brain_nodes.consult_fia_resources(cast(Any, state))

    assert out["responses"][0]["intent"] == IntentType.CONSULT_FIA_RESOURCES
    assert out.get("collection_used") == "en_fia_resources"

    assert captured_messages.get("model") == "gpt-4o-mini"

    # Ensure the FIA manual and vector doc both reached the prompt payload
    payload = captured_messages["input"][0]["content"]
    assert "FIA manual reference text" in payload
    assert "English FIA doc" in payload


def test_consult_fia_resources_uses_normal_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Defaults to gpt-4o when agentic strength remains normal."""

    class _FakeResponse:  # pylint: disable=too-few-public-methods
        usage = None
        output_text = "fia guidance"

    captured: dict[str, Any] = {}

    def _fake_create(**kwargs: Any) -> Any:  # noqa: ANN401
        captured["model"] = kwargs.get("model")
        return _FakeResponse()

    runtime.get_services().chroma = StubChromaPort(
        {
            "en_fia_resources": _FakeCollection(
                [("Localized", 0.2, {"name": "Localized", "source": "fia/local.md"})]
            )
        }
    )
    monkeypatch.setattr(brain_nodes.open_ai_client.responses, "create", _fake_create)
    monkeypatch.setattr(brain_nodes, "FIA_REFERENCE_CONTENT", "reference body")

    state: dict[str, Any] = {
        "transformed_query": "Explain FIA step 3",
        "user_chat_history": [],
        "user_response_language": "en",
        "query_language": "en",
        "agentic_strength": "normal",
    }

    out = brain_nodes.consult_fia_resources(cast(Any, state))

    assert out["responses"], "expected consult FIA response"
    assert captured.get("model") == "gpt-4o"


def test_consult_fia_resources_handles_missing_context(monkeypatch: pytest.MonkeyPatch) -> None:
    """Returns a helpful fallback when no FIA resources are available."""
    from bt_servant_engine.services.intents import fia_intents

    services = runtime.get_services()
    services.chroma = StubChromaPort({})
    monkeypatch.setattr(fia_intents, "FIA_REFERENCE_CONTENT", "")

    state: dict[str, Any] = {
        "transformed_query": "How do I translate the Bible?",
        "user_chat_history": [],
        "user_response_language": None,
        "query_language": "en",
    }

    out = brain_nodes.consult_fia_resources(cast(Any, state))

    response_text = out["responses"][0]["response"]
    assert "couldn't find any FIA resources" in response_text
    assert IntentType.CONSULT_FIA_RESOURCES == out["responses"][0]["intent"]
