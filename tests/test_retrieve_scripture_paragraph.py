"""Tests for retrieve-scripture paragraph formatting (no ch:vs labels)."""
# pylint: disable=missing-module-docstring,too-few-public-methods,missing-function-docstring,unused-argument,duplicate-code

import re
from typing import Any, cast

import pytest

import brain


class _StubParseResult:
    def __init__(self, output_parsed: Any):
        self.output_parsed = output_parsed
        self.usage = None


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


def test_retrieve_scripture_returns_paragraph_without_labels(monkeypatch: pytest.MonkeyPatch):
    # Arrange: stub selection parse; no explicit requested language
    def parse_stub(*args: Any, **kwargs: Any):  # noqa: ANN401 - test stub
        tf = kwargs.get("text_format")
        if tf is brain.PassageSelection:
            sel = brain.PassageSelection(
                selections=[
                    brain.PassageRef(
                        book="Genesis",
                        start_chapter=1,
                        start_verse=1,
                        end_chapter=1,
                        end_verse=3,
                    )
                ]
            )
            return _StubParseResult(sel)
        if tf is brain.ResponseLanguage:
            # No explicit requested language in message
            return _StubParseResult(brain.ResponseLanguage(language=brain.Language.OTHER))
        return _StubParseResult(None)

    monkeypatch.setattr(brain.open_ai_client.responses, "parse", parse_stub)

    state = _state_for("Please provide the text of Genesis 1:1-3")

    # Act
    out = brain.handle_retrieve_scripture(state)

    # Assert: structured scripture response with a flowing paragraph (no ch:vs labels, no newlines)
    item = (out.get("responses") or [])[0]
    resp = cast(dict, item["response"])
    assert resp.get("suppress_translation") is True
    segs = cast(list, resp.get("segments"))
    scripture_seg = next((s for s in segs if s.get("type") == "scripture"), None)
    assert scripture_seg is not None, "expected a scripture segment"
    body = cast(str, scripture_seg.get("text", ""))
    assert body, "expected non-empty scripture body"
    assert "\n" not in body, "scripture body should be a single paragraph with spaces"
    assert not re.search(r"\b\d+:\d+\b", body), "chapter:verse labels should be removed"
