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
