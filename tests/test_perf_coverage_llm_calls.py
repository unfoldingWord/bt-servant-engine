"""Ensure LLM calls attribute tokens to perf spans for key paths.

These tests stub OpenAI SDK calls to avoid network and to return `usage`
objects so that `utils.perf.add_tokens(...)` is invoked within active spans.
"""

from typing import Any, Iterable, Mapping, cast

import pytest

from bt_servant_engine.core.config import config
from bt_servant_engine.core.intents import IntentType, UserIntents
from bt_servant_engine.core.models import PassageRef, PassageSelection
from bt_servant_engine.core.ports import ChromaPort
from bt_servant_engine.services import brain_nodes, response_pipeline, runtime
from utils import perf

INTENT_INPUT_TOKENS = 50
INTENT_OUTPUT_TOKENS = 10
INTENT_TOTAL_TOKENS = 60
INTENT_CACHED_TOKENS = 5

SELECTION_INPUT_TOKENS = 30
SELECTION_OUTPUT_TOKENS = 5
SELECTION_TOTAL_TOKENS = 35
SELECTION_CACHED_TOKENS = 2

TRANSLATE_PROMPT_TOKENS = 25
TRANSLATE_COMPLETION_TOKENS = 10
TRANSLATE_TOTAL_TOKENS = 35
TRANSLATE_CACHED_TOKENS = 3

CHUNK_PROMPT_TOKENS = 20
CHUNK_COMPLETION_TOKENS = 8
CHUNK_TOTAL_TOKENS = 28
CHUNK_CACHED_TOKENS = 1

FIA_PROMPT_TOKENS = 45
FIA_COMPLETION_TOKENS = 12
FIA_TOTAL_TOKENS = 57
FIA_CACHED_TOKENS = 6

CHUNK_OVERFLOW_PADDING = 50
COMBINE_INPUT_TOKENS = 40
COMBINE_OUTPUT_TOKENS = 15
COMBINE_TOTAL_TOKENS = 55
COMBINE_CACHED_TOKENS = 4


class _FakeInputTokenDetails:  # pylint: disable=too-few-public-methods
    """Lightweight stand-in for usage.input_token_details."""

    def __init__(self, cached: int | None = None) -> None:
        self.cache_read_input_tokens = cached


class _FakeUsage:  # pylint: disable=too-few-public-methods
    """Fake usage payload to simulate SDK accounting."""

    def __init__(self, it: int, ot: int, tt: int | None = None, cached: int | None = None) -> None:
        self.input_tokens = it
        self.output_tokens = ot
        self.total_tokens = (it + ot) if tt is None else tt
        self.input_token_details = _FakeInputTokenDetails(cached)


def _find_span(report: dict[str, Any], name: str) -> dict[str, Any] | None:
    for s in cast(list[dict[str, Any]], report.get("spans", [])):
        if s.get("name") == name:
            return s
    return None


def test_determine_intents_span_has_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tokens from intent classification are attributed to its span."""
    tid = "trace-det-intents"
    perf.set_current_trace(tid)

    class _FakeResp:  # pylint: disable=too-few-public-methods
        usage = _FakeUsage(
            it=INTENT_INPUT_TOKENS,
            ot=INTENT_OUTPUT_TOKENS,
            tt=INTENT_TOTAL_TOKENS,
            cached=INTENT_CACHED_TOKENS,
        )
        output_parsed = UserIntents(intents=[IntentType.GET_PASSAGE_SUMMARY])

    def _fake_parse(**_kwargs: Any) -> Any:  # noqa: ARG001 - signature must accept kwargs
        return _FakeResp()

    monkeypatch.setattr(brain_nodes.open_ai_client.responses, "parse", _fake_parse)

    with perf.time_block("brain:determine_intents_node"):
        out = brain_nodes.determine_intents({"transformed_query": "dummy"})

    # Sanity: function returns parsed intents
    assert out["user_intents"] and out["user_intents"][0] == IntentType.GET_PASSAGE_SUMMARY

    report = perf.summarize_report(tid)
    span = _find_span(report, "brain:determine_intents_node")
    assert span is not None, "expected a span for determine_intents_node"
    assert span.get("input_tokens_expended") == INTENT_INPUT_TOKENS
    assert span.get("output_tokens_expended") == INTENT_OUTPUT_TOKENS
    assert span.get("total_tokens_expended") == INTENT_TOTAL_TOKENS


def test_selection_helper_tokens_roll_into_parent_span(monkeypatch: pytest.MonkeyPatch) -> None:
    """Selection helper tokens roll up into the current node span."""
    tid = "trace-selection-helper"
    perf.set_current_trace(tid)

    # Fake a parsed selection for a simple reference
    fake_selection = PassageSelection(
        selections=[
            PassageRef(
                book="John",
                start_chapter=3,
                start_verse=16,
                end_chapter=3,
                end_verse=18,
            )
        ]
    )

    class _FakeResp:  # pylint: disable=too-few-public-methods
        usage = _FakeUsage(
            it=SELECTION_INPUT_TOKENS,
            ot=SELECTION_OUTPUT_TOKENS,
            tt=SELECTION_TOTAL_TOKENS,
            cached=SELECTION_CACHED_TOKENS,
        )
        output_parsed = fake_selection

    def _fake_parse(**_kwargs: Any) -> Any:  # noqa: ARG001 - signature must accept kwargs
        return _FakeResp()

    monkeypatch.setattr(brain_nodes.open_ai_client.responses, "parse", _fake_parse)

    # Open a span matching the keywords node; helper should add tokens to it
    with perf.time_block("brain:handle_get_passage_keywords_node"):
        book, ranges, err = brain_nodes.resolve_selection_for_single_book(
            "John 3:16-18",
            "en",
        )

    assert err is None
    assert book == "John" and ranges is not None

    report = perf.summarize_report(tid)
    span = _find_span(report, "brain:handle_get_passage_keywords_node")
    assert span is not None, "expected a span for handle_get_passage_keywords_node"
    assert span.get("input_tokens_expended") == SELECTION_INPUT_TOKENS
    assert span.get("output_tokens_expended") == SELECTION_OUTPUT_TOKENS
    assert span.get("total_tokens_expended") == SELECTION_TOTAL_TOKENS


def test_translate_responses_span_has_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    """Translate LLM usage contributes tokens to translate_responses span."""
    tid = "trace-translate-responses"
    perf.set_current_trace(tid)

    # Stub unused (kept for backward compatibility with test structure)
    class _FakeCombine:  # pylint: disable=too-few-public-methods
        usage = _FakeUsage(
            it=COMBINE_INPUT_TOKENS,
            ot=COMBINE_OUTPUT_TOKENS,
            tt=COMBINE_TOTAL_TOKENS,
            cached=COMBINE_CACHED_TOKENS,
        )
        output_text = "Combined response"

    def _fake_resp_create(**_kwargs: Any) -> Any:  # noqa: ARG001
        return _FakeCombine()

    # Stub translate_text Chat Completions.create
    class _Msg:  # pylint: disable=too-few-public-methods
        def __init__(self, content: str) -> None:
            self.content = content

    class _Choice:  # pylint: disable=too-few-public-methods
        def __init__(self, content: str) -> None:
            self.message = _Msg(content)

    class _FakeChat:  # pylint: disable=too-few-public-methods
        def __init__(self) -> None:
            # Chat Completions usage exposes prompt_tokens/completion_tokens
            class _U:  # pylint: disable=too-few-public-methods
                def __init__(self) -> None:
                    self.prompt_tokens = TRANSLATE_PROMPT_TOKENS
                    self.completion_tokens = TRANSLATE_COMPLETION_TOKENS
                    self.total_tokens = TRANSLATE_TOTAL_TOKENS

                    class _PTD:  # pylint: disable=too-few-public-methods
                        def __init__(self) -> None:
                            self.cached_tokens = TRANSLATE_CACHED_TOKENS

                    self.prompt_tokens_details = _PTD()

            self.usage = _U()
            self.choices = [_Choice("Hola")]  # translated content

    def _fake_chat_create(**_kwargs: Any) -> Any:  # noqa: ARG001
        return _FakeChat()

    # Force translation by reporting response language != target language

    monkeypatch.setattr(response_pipeline, "detect_language_impl", lambda _client, _t, **_k: "en")
    monkeypatch.setattr(brain_nodes.open_ai_client.responses, "create", _fake_resp_create)
    monkeypatch.setattr(brain_nodes.open_ai_client.chat.completions, "create", _fake_chat_create)

    state = {
        "responses": [
            {"intent": "x", "response": "Hello"},
        ],
        "user_query": "irrelevant",
        "user_chat_history": [],
        "user_response_language": "es",  # target
        "query_language": "en",
    }

    with perf.time_block("brain:translate_responses_node"):
        out = brain_nodes.translate_responses(cast(Any, state))

    assert out["translated_responses"], "expected translated responses"
    report = perf.summarize_report(tid)
    span = _find_span(report, "brain:translate_responses_node")
    assert span is not None, "expected a span for translate_responses_node"
    # With sequential processing, expect only translate tokens (no combine)
    # translate call: 25 input + 10 output = 35 total, 3 cached
    assert span.get("input_tokens_expended") == TRANSLATE_PROMPT_TOKENS
    assert span.get("output_tokens_expended") == TRANSLATE_COMPLETION_TOKENS
    assert span.get("total_tokens_expended") == TRANSLATE_TOTAL_TOKENS


def test_chunk_message_span_has_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    """LLM chunking tokens contribute to chunk_message span."""
    tid = "trace-chunk-message"
    perf.set_current_trace(tid)

    class _Msg:  # pylint: disable=too-few-public-methods
        def __init__(self, content: str) -> None:
            self.content = content

    class _Choice:  # pylint: disable=too-few-public-methods
        def __init__(self, content: str) -> None:
            self.message = _Msg(content)

    class _FakeChat:  # pylint: disable=too-few-public-methods
        def __init__(self) -> None:
            class _U:  # pylint: disable=too-few-public-methods
                def __init__(self) -> None:
                    self.prompt_tokens = CHUNK_PROMPT_TOKENS
                    self.completion_tokens = CHUNK_COMPLETION_TOKENS
                    self.total_tokens = CHUNK_TOTAL_TOKENS

                    class _PTD:  # pylint: disable=too-few-public-methods
                        def __init__(self) -> None:
                            self.cached_tokens = CHUNK_CACHED_TOKENS

                    self.prompt_tokens_details = _PTD()

            self.usage = _U()
            # Return valid JSON list to avoid fallback path
            self.choices = [_Choice('["a","b"]')]

    def _fake_chat_create(**_kwargs: Any) -> Any:  # noqa: ARG001
        return _FakeChat()

    monkeypatch.setattr(brain_nodes.open_ai_client.chat.completions, "create", _fake_chat_create)

    long_text = "x" * (config.MAX_RESPONSE_CHUNK_SIZE + CHUNK_OVERFLOW_PADDING)
    state = {
        "translated_responses": [long_text],
    }

    with perf.time_block("brain:chunk_message_node"):
        out = brain_nodes.chunk_message(cast(Any, state))

    assert out["translated_responses"], "expected chunked responses"
    report = perf.summarize_report(tid)
    span = _find_span(report, "brain:chunk_message_node")
    assert span is not None, "expected a span for chunk_message_node"
    assert span.get("input_tokens_expended") == CHUNK_PROMPT_TOKENS
    assert span.get("output_tokens_expended") == CHUNK_COMPLETION_TOKENS
    assert span.get("total_tokens_expended") == CHUNK_TOTAL_TOKENS


def test_consult_fia_resources_span_has_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    """Consult FIA resources should attribute its LLM usage to the node span."""

    tid = "trace-consult-fia"
    perf.set_current_trace(tid)

    class _FakeCollection:  # pylint: disable=too-few-public-methods
        def __init__(self) -> None:
            self._docs = [
                (
                    "FIA localized note",
                    0.05,
                    {"name": "Localized FIA", "source": "fia/localized.md"},
                ),
            ]

        def query(self, **_kwargs: Any) -> dict[str, Any]:  # noqa: ANN401
            """Return the fake query payload for the test."""
            documents = [doc for doc, _dist, _meta in self._docs]
            distances = [dist for _doc, dist, _meta in self._docs]
            metadatas = [meta for _doc, _dist, meta in self._docs]
            return {
                "documents": [documents],
                "distances": [distances],
                "metadatas": [metadatas],
            }

    class _FakeResponse:  # pylint: disable=too-few-public-methods
        usage = _FakeUsage(
            it=FIA_PROMPT_TOKENS,
            ot=FIA_COMPLETION_TOKENS,
            tt=FIA_TOTAL_TOKENS,
            cached=FIA_CACHED_TOKENS,
        )
        output_text = "fia response"

    class _StubChromaPort(ChromaPort):
        def __init__(self) -> None:
            self._collection = _FakeCollection()

        def get_collection(self, name: str) -> _FakeCollection | None:  # noqa: D401
            del name
            return self._collection

        def list_collections(self) -> list[str]:
            return ["fia"]

        def create_collection(self, name: str) -> None:  # noqa: D401
            del name

        def delete_collection(self, name: str) -> None:  # noqa: D401
            del name

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
            del source, dest
            raise RuntimeError("Not supported")

        def max_numeric_id(self, name: str) -> int:
            del name
            return 0

    runtime.get_services().chroma = _StubChromaPort()
    monkeypatch.setattr(brain_nodes, "FIA_REFERENCE_CONTENT", "manual snippet")
    monkeypatch.setattr(
        brain_nodes.open_ai_client.responses, "create", lambda **_k: _FakeResponse()
    )

    state = {
        "transformed_query": "How do I translate the Bible?",
        "user_chat_history": [],
        "user_response_language": "en",
        "query_language": "en",
    }

    with perf.time_block("brain:consult_fia_resources_node"):
        out = brain_nodes.consult_fia_resources(cast(Any, state))

    assert out["responses"], "expected a response payload"

    report = perf.summarize_report(tid)
    span = _find_span(report, "brain:consult_fia_resources_node")
    assert span is not None, "expected a span for consult_fia_resources_node"
    assert span.get("input_tokens_expended") == FIA_PROMPT_TOKENS
    assert span.get("output_tokens_expended") == FIA_COMPLETION_TOKENS
    assert span.get("total_tokens_expended") == FIA_TOTAL_TOKENS
