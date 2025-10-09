"""Unit tests for graph pipeline helper functions."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List, cast
from openai import OpenAI, OpenAIError

from bt_servant_engine.core.intents import IntentType
from bt_servant_engine.services import graph_pipeline as gp

# pylint: disable=missing-function-docstring,missing-class-docstring,too-few-public-methods


class DummyCollection:
    def __init__(self, results: Dict[str, List[List[Any]]]):
        self._results = results

    def query(self, **kwargs):  # noqa: D401
        del kwargs
        return self._results


def test_query_vector_db_filters_by_relevance() -> None:
    relevant_results: Dict[str, List[List[Any]]] = {
        "documents": [["doc-a", "doc-b"]],
        "distances": [[0.1, 0.7]],  # cosine similarity -> 0.9 and 0.3
        "metadatas": [
            [
                {"name": "ResA", "source": "SrcA", "_merged_from": "uw_notes"},
                {"name": "ResB", "source": "SrcB", "_merged_from": "uw_dictionary"},
            ]
        ],
    }
    collections = {
        "missing": None,
        "existing": DummyCollection(relevant_results),
    }

    def get_collection(name: str):
        return collections.get(name)

    result = gp.query_vector_db(
        "query text",
        ["missing", "existing"],
        get_collection,
        "fallback",
        relevance_cutoff=0.5,
    )
    assert result["docs"] == [
        {
            "collection_name": "existing",
            "resource_name": "ResA",
            "source": "SrcA",
            "document_text": "doc-a",
            "metadata": {"name": "ResA", "source": "SrcA", "_merged_from": "uw_notes"},
        }
    ]


class DummyResponsesClient:
    def __init__(self, text: str):
        self.text = text
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        usage = SimpleNamespace(input_tokens=10, output_tokens=5, total_tokens=15)
        return SimpleNamespace(output_text=self.text, usage=usage)


class DummyClient:
    def __init__(self, text: str):
        self.responses = DummyResponsesClient(text)


def test_query_open_ai_returns_combined_response() -> None:
    client = DummyClient("final answer")
    docs = [
        {"resource_name": "Doc1", "source": "src", "document_text": "text"},
        {"resource_name": "Doc2", "source": "src", "document_text": "text"},
    ]
    chat_history = [{"role": "user", "content": "hi"}]
    tokens_recorded: list[tuple] = []

    result = gp.query_open_ai(
        cast(OpenAI, client),
        docs,
        "question?",
        chat_history,
        lambda strength, allow_low, allow_very_low: "gpt-4o-mini",  # noqa: ARG005
        lambda usage: None,
        lambda *args, **kwargs: tokens_recorded.append((args, kwargs)),
        agentic_strength="low",
        boilerplate_features_message="features",
    )

    assert result["responses"][0]["intent"] is IntentType.GET_BIBLE_TRANSLATION_ASSISTANCE
    assert result["responses"][0]["response"] == "final answer"
    assert tokens_recorded, "expected token accounting"
    assert client.responses.calls, "expected responses.create to be invoked"


def test_query_open_ai_handles_no_docs() -> None:
    client = DummyClient("unused")
    fallback = gp.query_open_ai(
        cast(OpenAI, client),
        [],
        "question?",
        [],
        lambda *args, **kwargs: "gpt-4o",  # noqa: ARG005
        lambda usage: None,
        lambda *args, **kwargs: None,
        agentic_strength="normal",
        boilerplate_features_message="feature list",
    )
    response_text = fallback["responses"][0]["response"]
    assert "feature list" in response_text


def test_query_open_ai_surfaces_openai_errors() -> None:
    class ErrorClient:
        class Responses:
            @staticmethod
            def create(*args, **kwargs):
                del args, kwargs
                raise OpenAIError("boom")

        responses = Responses()

    err = gp.query_open_ai(
        cast(OpenAI, ErrorClient()),
        [{"resource_name": "Doc1", "source": "src", "document_text": "text"}],
        "question?",
        [],
        lambda *args, **kwargs: "gpt-4o",  # noqa: ARG005
        lambda usage: None,
        lambda *args, **kwargs: None,
        agentic_strength="normal",
        boilerplate_features_message="feature list",
    )
    assert "Let Ian know" in err["responses"][0]["response"]
