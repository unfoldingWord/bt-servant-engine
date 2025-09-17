"""Ensure LLM calls attribute tokens to perf spans for key paths.

These tests stub OpenAI SDK calls to avoid network and to return `usage`
objects so that `utils.perf.add_tokens(...)` is invoked within active spans.
"""

from typing import Any, cast

import pytest

from utils import perf
import brain


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
        usage = _FakeUsage(it=50, ot=10, tt=60, cached=5)
        output_parsed = brain.UserIntents(
            intents=[brain.IntentType.GET_PASSAGE_SUMMARY]
        )

    def _fake_parse(**_kwargs: Any) -> Any:  # noqa: ARG001 - signature must accept kwargs
        return _FakeResp()

    monkeypatch.setattr(brain.open_ai_client.responses, "parse", _fake_parse)

    with perf.time_block("brain:determine_intents_node"):
        out = brain.determine_intents({"transformed_query": "dummy"})

    # Sanity: function returns parsed intents
    assert out["user_intents"] and out["user_intents"][0] == brain.IntentType.GET_PASSAGE_SUMMARY

    report = perf.summarize_report(tid)
    span = _find_span(report, "brain:determine_intents_node")
    assert span is not None, "expected a span for determine_intents_node"
    assert span.get("input_tokens_expended") == 50
    assert span.get("output_tokens_expended") == 10
    assert span.get("total_tokens_expended") == 60


def test_selection_helper_tokens_roll_into_parent_span(monkeypatch: pytest.MonkeyPatch) -> None:
    """Selection helper tokens roll up into the current node span."""
    tid = "trace-selection-helper"
    perf.set_current_trace(tid)

    # Fake a parsed selection for a simple reference
    fake_selection = brain.PassageSelection(
        selections=[
            brain.PassageRef(
                book="John",
                start_chapter=3,
                start_verse=16,
                end_chapter=3,
                end_verse=18,
            )
        ]
    )

    class _FakeResp:  # pylint: disable=too-few-public-methods
        usage = _FakeUsage(it=30, ot=5, tt=35, cached=2)
        output_parsed = fake_selection

    def _fake_parse(**_kwargs: Any) -> Any:  # noqa: ARG001 - signature must accept kwargs
        return _FakeResp()

    monkeypatch.setattr(brain.open_ai_client.responses, "parse", _fake_parse)

    # Open a span matching the keywords node; helper should add tokens to it
    with perf.time_block("brain:handle_get_passage_keywords_node"):
        # Accessing a protected helper is acceptable in tests for coverage
        # of token attribution in the selection phase.
        book, ranges, err = brain._resolve_selection_for_single_book(  # pylint: disable=protected-access
            "John 3:16-18",
            "en",
        )

    assert err is None
    assert book == "John" and ranges is not None

    report = perf.summarize_report(tid)
    span = _find_span(report, "brain:handle_get_passage_keywords_node")
    assert span is not None, "expected a span for handle_get_passage_keywords_node"
    assert span.get("input_tokens_expended") == 30
    assert span.get("output_tokens_expended") == 5
    assert span.get("total_tokens_expended") == 35


def test_translate_responses_span_has_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    """combine + translate LLM usage contributes tokens to translate_responses span."""
    tid = "trace-translate-responses"
    perf.set_current_trace(tid)

    # Stub combine_responses Responses.create
    class _FakeCombine:  # pylint: disable=too-few-public-methods
        usage = _FakeUsage(it=40, ot=15, tt=55, cached=4)
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
                    self.prompt_tokens = 25
                    self.completion_tokens = 10
                    self.total_tokens = 35
                    class _PTD:  # pylint: disable=too-few-public-methods
                        def __init__(self) -> None:
                            self.cached_tokens = 3
                    self.prompt_tokens_details = _PTD()
            self.usage = _U()
            self.choices = [_Choice("Hola")]  # translated content

    def _fake_chat_create(**_kwargs: Any) -> Any:  # noqa: ARG001
        return _FakeChat()

    # Force translation by reporting response language != target language
    monkeypatch.setattr(brain, "detect_language", lambda _t, **_k: "en")
    monkeypatch.setattr(brain.open_ai_client.responses, "create", _fake_resp_create)
    monkeypatch.setattr(brain.open_ai_client.chat.completions, "create", _fake_chat_create)

    state = {
        "responses": [
            {"intent": "x", "response": "Hello"},
            {"intent": "y", "response": "World"},
        ],
        "user_query": "irrelevant",
        "user_chat_history": [],
        "user_response_language": "es",  # target
        "query_language": "en",
    }

    with perf.time_block("brain:translate_responses_node"):
        out = brain.translate_responses(cast(Any, state))

    assert out["translated_responses"], "expected translated responses"
    report = perf.summarize_report(tid)
    span = _find_span(report, "brain:translate_responses_node")
    assert span is not None, "expected a span for translate_responses_node"
    # Expect at least the sum from combine (40/15/55) plus translate (25/10/35)
    assert span.get("input_tokens_expended") == 65
    assert span.get("output_tokens_expended") == 25
    assert span.get("total_tokens_expended") == 90


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
                    self.prompt_tokens = 20
                    self.completion_tokens = 8
                    self.total_tokens = 28
                    class _PTD:  # pylint: disable=too-few-public-methods
                        def __init__(self) -> None:
                            self.cached_tokens = 1
                    self.prompt_tokens_details = _PTD()
            self.usage = _U()
            # Return valid JSON list to avoid fallback path
            self.choices = [_Choice('["a","b"]')]

    def _fake_chat_create(**_kwargs: Any) -> Any:  # noqa: ARG001
        return _FakeChat()

    monkeypatch.setattr(brain.open_ai_client.chat.completions, "create", _fake_chat_create)

    long_text = "x" * (brain.config.MAX_META_TEXT_LENGTH + 50)
    state = {
        "translated_responses": [long_text],
    }

    with perf.time_block("brain:chunk_message_node"):
        out = brain.chunk_message(cast(Any, state))

    assert out["translated_responses"], "expected chunked responses"
    report = perf.summarize_report(tid)
    span = _find_span(report, "brain:chunk_message_node")
    assert span is not None, "expected a span for chunk_message_node"
    assert span.get("input_tokens_expended") == 20
    assert span.get("output_tokens_expended") == 8
    assert span.get("total_tokens_expended") == 28


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
        usage = _FakeUsage(it=45, ot=12, tt=57, cached=6)
        output_text = "fia response"

    monkeypatch.setattr(brain, "get_chroma_collection", lambda _name: _FakeCollection())
    monkeypatch.setattr(brain, "FIA_REFERENCE_CONTENT", "manual snippet")
    monkeypatch.setattr(brain.open_ai_client.responses, "create", lambda **_k: _FakeResponse())

    state = {
        "transformed_query": "How do I translate the Bible?",
        "user_chat_history": [],
        "user_response_language": "en",
        "query_language": "en",
    }

    with perf.time_block("brain:consult_fia_resources_node"):
        out = brain.consult_fia_resources(cast(Any, state))

    assert out["responses"], "expected a response payload"

    report = perf.summarize_report(tid)
    span = _find_span(report, "brain:consult_fia_resources_node")
    assert span is not None, "expected a span for consult_fia_resources_node"
    assert span.get("input_tokens_expended") == 45
    assert span.get("output_tokens_expended") == 12
    assert span.get("total_tokens_expended") == 57
