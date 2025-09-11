"""Unit tests for translate-scripture intent permutations."""
# pylint: disable=missing-module-docstring,line-too-long,too-few-public-methods,unused-argument,missing-function-docstring

from typing import Any, cast

import pytest

import brain


class _StubParseResult:
    def __init__(self, output_parsed: Any):
        self.output_parsed = output_parsed
        self.usage = None


def _make_parse_stub(current_query: str):
    """Return a stub for open_ai_client.responses.parse that inspects text_format."""

    def _parse_stub(*args: Any, **kwargs: Any):  # noqa: ANN401 - test stub
        text_format = kwargs.get("text_format")
        # Target language detection calls set instructions and text_format=ResponseLanguage
        if text_format is brain.ResponseLanguage:
            q = current_query.lower()
            if "dutch" in q:
                return _StubParseResult(brain.ResponseLanguage(language=brain.Language.DUTCH))
            if "chinese" in q:
                return _StubParseResult(brain.ResponseLanguage(language=brain.Language.MANDARIN))
            # For unsupported languages like Italian or when no target present
            return _StubParseResult(brain.ResponseLanguage(language=brain.Language.OTHER))

        # Passage selection parse: text_format=PassageSelection
        if text_format is brain.PassageSelection:
            q = current_query.lower()
            if "enoch" in q:
                # Return a selection using an unsupported book name to trigger error
                sel = brain.PassageSelection(selections=[
                    brain.PassageRef(book="Enoch", start_chapter=1, start_verse=None, end_chapter=None, end_verse=None)
                ])
                return _StubParseResult(sel)
            # Default: Genesis 1:1-3
            sel = brain.PassageSelection(selections=[
                brain.PassageRef(book="Genesis", start_chapter=1, start_verse=1, end_chapter=1, end_verse=3)
            ])
            return _StubParseResult(sel)

        # Fallback
        return _StubParseResult(None)

    return _parse_stub


def _state_for(query: str) -> brain.BrainState:
    return cast(brain.BrainState, {
        "user_id": "test-user",
        "user_query": query,
        "transformed_query": query,
        "query_language": "en",
        "user_response_language": "",
        "responses": [],
        "user_chat_history": [],
    })


@pytest.mark.parametrize(
    "query, expect_substring",
    [
        ("translate gen 1:1-3", "unsupported language"),
        ("translate gen 1:1 into dutch", "into Dutch is currently not supported"),
        ("translate gen 1:1-3 into italian", "into Italian is currently not supported"),
    ],
)
def test_translate_scripture_guidance_messages(monkeypatch: pytest.MonkeyPatch, query: str, expect_substring: str):
    # Arrange: stub out Responses API parse calls
    monkeypatch.setattr(brain.open_ai_client.responses, "parse", _make_parse_stub(query))

    state = _state_for(query)

    # Act
    out = brain.handle_translate_scripture(state)

    # Assert
    items = out.get("responses") or []
    assert items, "expected a response item"
    msg = cast(str, items[0]["response"])  # guidance returns a string
    assert expect_substring in msg
    # Ensure the supported languages list is present once
    for lang in ("English", "Arabic", "French", "Spanish", "Hindi", "Russian", "Indonesian", "Swahili", "Portuguese", "Mandarin", "Dutch"):
        assert lang in msg


@pytest.mark.parametrize(
    "query",
    [
        "translate enoch 1 into chinese",
        "translate enoch 1",
    ],
)
def test_translate_scripture_unsupported_book_returns_selection_error(monkeypatch: pytest.MonkeyPatch, query: str):
    # Arrange
    monkeypatch.setattr(brain.open_ai_client.responses, "parse", _make_parse_stub(query))
    state = _state_for(query)

    # Act
    out = brain.handle_translate_scripture(state)

    # Assert: should return selection error about unsupported book, not guidance
    items = out.get("responses") or []
    assert items, "expected a response item"
    msg = cast(str, items[0]["response"])
    assert "not recognized" in msg and "Enoch" in msg
