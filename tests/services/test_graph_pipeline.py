"""Unit tests for graph pipeline helper functions."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List, cast
from openai import OpenAI, OpenAIError

from bt_servant_engine.core.intents import IntentType
from bt_servant_engine.services import graph_pipeline as gp


def _make_collection(results: Dict[str, List[List[Any]]]) -> SimpleNamespace:
    """Return a SimpleNamespace with a query method returning canned results."""

    def query(**kwargs) -> Dict[str, List[List[Any]]]:
        del kwargs
        return results

    return SimpleNamespace(query=query)


def test_query_vector_db_filters_by_relevance() -> None:
    """Vector DB query filters documents below the relevance cutoff."""
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
        "existing": _make_collection(relevant_results),
    }

    def get_collection(name: str):
        return collections.get(name)

    result = gp.query_vector_db(
        "query text",
        ["missing", "existing"],
        get_collection,
        "fallback",
        config=gp.VectorQueryConfig(relevance_cutoff=0.5),
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


def _make_responses_client(text: str) -> tuple[SimpleNamespace, list[dict]]:
    """Helper returning a responses client stub and the recorded call list."""
    calls: list[dict] = []

    def create(**kwargs):
        calls.append(kwargs)
        usage = SimpleNamespace(input_tokens=10, output_tokens=5, total_tokens=15)
        return SimpleNamespace(output_text=text, usage=usage)

    return SimpleNamespace(create=create), calls


def test_query_open_ai_returns_combined_response() -> None:
    """OpenAI query combines docs and chat history into a response."""
    responses_client, recorded_calls = _make_responses_client("final answer")
    client = SimpleNamespace(responses=responses_client)
    docs = [
        {"resource_name": "Doc1", "source": "src", "document_text": "text"},
        {"resource_name": "Doc2", "source": "src", "document_text": "text"},
    ]
    chat_history = [{"role": "user", "content": "hi"}]
    tokens_recorded: list[tuple] = []

    def choose_model(strength: str, allow_low: bool, allow_very_low: bool) -> str:
        del strength, allow_low, allow_very_low
        return "gpt-4o-mini"

    deps = gp.OpenAIQueryDependencies(
        model_for_agentic_strength=choose_model,
        extract_cached_input_tokens=lambda usage: None,
        add_tokens=lambda *args, **kwargs: tokens_recorded.append((args, kwargs)),
    )
    payload = gp.OpenAIQueryPayload(
        docs=docs,
        transformed_query="question?",
        chat_history=chat_history,
        agentic_strength="low",
        boilerplate_features_message="features",
    )
    result = gp.query_open_ai(
        cast(OpenAI, client),
        payload,
        deps,
    )

    first_response = result["responses"][0]
    assert first_response["intent"] is IntentType.GET_BIBLE_TRANSLATION_ASSISTANCE
    assert first_response["response"] == "final answer"
    assert tokens_recorded, "expected token accounting"
    assert recorded_calls, "expected responses.create to be invoked"


def test_query_open_ai_handles_no_docs() -> None:
    """OpenAI query falls back to boilerplate when no docs are available."""
    responses_client, _ = _make_responses_client("unused")
    client = SimpleNamespace(responses=responses_client)
    deps = gp.OpenAIQueryDependencies(
        model_for_agentic_strength=lambda *args, **kwargs: "gpt-4o",  # noqa: ARG005
        extract_cached_input_tokens=lambda usage: None,
        add_tokens=lambda *args, **kwargs: None,
    )
    payload = gp.OpenAIQueryPayload(
        docs=[],
        transformed_query="question?",
        chat_history=[],
        agentic_strength="normal",
        boilerplate_features_message="feature list",
    )
    fallback = gp.query_open_ai(
        cast(OpenAI, client),
        payload,
        deps,
    )
    response_text = fallback["responses"][0]["response"]
    assert "feature list" in response_text


def test_query_open_ai_surfaces_openai_errors() -> None:
    """OpenAI query returns a friendly error for API failures."""
    def create_error(*args, **kwargs):
        del args, kwargs
        raise OpenAIError("boom")

    client = SimpleNamespace(responses=SimpleNamespace(create=create_error))

    deps = gp.OpenAIQueryDependencies(
        model_for_agentic_strength=lambda *args, **kwargs: "gpt-4o",  # noqa: ARG005
        extract_cached_input_tokens=lambda usage: None,
        add_tokens=lambda *args, **kwargs: None,
    )
    payload = gp.OpenAIQueryPayload(
        docs=[{"resource_name": "Doc1", "source": "src", "document_text": "text"}],
        transformed_query="question?",
        chat_history=[],
        agentic_strength="normal",
        boilerplate_features_message="feature list",
    )
    err = gp.query_open_ai(cast(OpenAI, client), payload, deps)
    assert "Let Ian know" in err["responses"][0]["response"]
