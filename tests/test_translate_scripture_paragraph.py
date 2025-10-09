"""Ensure translate-scripture returns a single paragraph (no newlines)."""
# pylint: disable=missing-module-docstring,too-few-public-methods,missing-function-docstring,duplicate-code

from typing import Any, cast

import pytest

import brain
from bt_servant_engine.core.models import PassageRef, PassageSelection
from bt_servant_engine.core.language import Language, ResponseLanguage, TranslatedPassage


class _StubParseResult:
    def __init__(self, output_parsed: Any):
        self.output_parsed = output_parsed
        self.usage = None


def _state_for(query: str) -> brain.BrainState:
    return cast(
        brain.BrainState,
        {
            "user_id": "test-user",
            "user_query": query,
            "transformed_query": query,
            "query_language": "en",
            "user_response_language": "",
            "responses": [],
            "user_chat_history": [],
        },
    )


def test_translate_scripture_normalizes_whitespace(monkeypatch: pytest.MonkeyPatch):
    # Arrange: supported target (Dutch). Return a TranslatedPassage with embedded newlines
    query = "translate genesis 1:1-2 into dutch"

    def parse_stub(*_args: Any, **kwargs: Any):  # noqa: ANN401 - test stub
        tf = kwargs.get("text_format")
        if tf is ResponseLanguage:
            return _StubParseResult(ResponseLanguage(language=Language.DUTCH))
        if tf is PassageSelection:
            sel = PassageSelection(
                selections=[
                    PassageRef(
                        book="Genesis",
                        start_chapter=1,
                        start_verse=1,
                        end_chapter=1,
                        end_verse=2,
                    )
                ]
            )
            return _StubParseResult(sel)
        if tf is TranslatedPassage:
            tp = TranslatedPassage(
                header_book="Genesis",
                header_suffix="1:1-2",
                body=("In den beginne\n\nschiep God\nde hemel en de aarde."),
                content_language=Language.DUTCH,
            )
            return _StubParseResult(tp)
        return _StubParseResult(None)

    monkeypatch.setattr(brain.open_ai_client.responses, "parse", parse_stub)

    state = _state_for(query)

    # Act
    out = brain.handle_translate_scripture(state)

    # Assert: scripture segment has no newlines
    item = (out.get("responses") or [])[0]
    resp = cast(dict, item["response"])
    segs = cast(list, resp.get("segments"))
    scripture_seg = next((s for s in segs if s.get("type") == "scripture"), None)
    assert scripture_seg is not None
    body = cast(str, scripture_seg.get("text", ""))
    assert body and "\n" not in body
