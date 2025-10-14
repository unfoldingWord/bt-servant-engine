"""Unit tests for translate-scripture intent permutations."""
# pylint: disable=missing-module-docstring,line-too-long,too-few-public-methods,unused-argument,missing-function-docstring

from typing import Any, cast

import pytest

from bt_servant_engine.core.language import Language, ResponseLanguage, TranslatedPassage
from bt_servant_engine.core.models import PassageRef, PassageSelection
from bt_servant_engine.services import brain_nodes
from bt_servant_engine.services.brain_orchestrator import BrainState


class _StubParseResult:
    def __init__(self, output_parsed: Any):
        self.output_parsed = output_parsed
        self.usage = None


def _make_parse_stub(current_query: str):
    """Return a stub for open_ai_client.responses.parse that inspects text_format."""

    def _parse_stub(*args: Any, **kwargs: Any):  # noqa: ANN401 - test stub
        text_format = kwargs.get("text_format")
        # Target language detection calls set instructions and text_format=ResponseLanguage
        if text_format is ResponseLanguage:
            q = current_query.lower()
            if "dutch" in q:
                return _StubParseResult(ResponseLanguage(language=Language.DUTCH))
            if "chinese" in q:
                return _StubParseResult(ResponseLanguage(language=Language.MANDARIN))
            # For unsupported languages like Italian or when no target present
            return _StubParseResult(ResponseLanguage(language=Language.OTHER))

        # Passage selection parse: text_format=PassageSelection
        if text_format is PassageSelection:
            q = current_query.lower()
            if "enoch" in q:
                # Return a selection using an unsupported book name to trigger error
                sel = PassageSelection(
                    selections=[
                        PassageRef(
                            book="Enoch",
                            start_chapter=1,
                            start_verse=None,
                            end_chapter=None,
                            end_verse=None,
                        )
                    ]
                )
                return _StubParseResult(sel)
            # Default: Genesis 1:1-3
            sel = PassageSelection(
                selections=[
                    PassageRef(
                        book="Genesis", start_chapter=1, start_verse=1, end_chapter=1, end_verse=3
                    )
                ]
            )
            return _StubParseResult(sel)

        # Fallback
        return _StubParseResult(None)

    return _parse_stub


def _state_for(query: str) -> BrainState:
    return cast(
        BrainState,
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


def test_translate_scripture_translates_with_supported_target(monkeypatch: pytest.MonkeyPatch):
    # Arrange: supported target (Dutch), stub both selection parse and translation parse
    query = "translate gen 1:1 into dutch"

    def parse_stub(*args: Any, **kwargs: Any):  # noqa: ANN401 - test stub
        tf = kwargs.get("text_format")
        if tf is ResponseLanguage:
            return _StubParseResult(ResponseLanguage(language=Language.DUTCH))
        if tf is PassageSelection:
            sel = PassageSelection(
                selections=[
                    PassageRef(
                        book="Genesis", start_chapter=1, start_verse=1, end_chapter=1, end_verse=1
                    )
                ]
            )
            return _StubParseResult(sel)
        if tf is TranslatedPassage:
            tp = TranslatedPassage(
                header_book="Genesis",
                header_suffix="1:1",
                body="In den beginne...",
                content_language=Language.DUTCH,
                follow_up_question="Wilt u dat ik de volgende 5 verzen van Genesis vertaal?",
            )
            return _StubParseResult(tp)
        return _StubParseResult(None)

    monkeypatch.setattr(brain_nodes.open_ai_client.responses, "parse", parse_stub)
    state = _state_for(query)

    # Act
    out = brain_nodes.handle_translate_scripture(state)

    # Assert: structured scripture response
    item = (out.get("responses") or [])[0]
    resp = cast(dict, item["response"])
    assert resp.get("suppress_translation") is True
    assert resp.get("content_language") == "nl"
    segs = cast(list, resp.get("segments"))
    assert any(s.get("type") == "scripture" for s in segs)


@pytest.mark.parametrize(
    "query",
    [
        "translate enoch 1 into chinese",
        "translate enoch 1",
    ],
)
def test_translate_scripture_unsupported_book_returns_selection_error(
    monkeypatch: pytest.MonkeyPatch, query: str
):
    # Arrange
    monkeypatch.setattr(brain_nodes.open_ai_client.responses, "parse", _make_parse_stub(query))
    state = _state_for(query)

    # Act
    out = brain_nodes.handle_translate_scripture(state)

    # Assert: should return selection error about unsupported book, not guidance
    items = out.get("responses") or []
    assert items, "expected a response item"
    msg = cast(str, items[0]["response"])
    assert "not recognized" in msg and "Enoch" in msg


def test_translate_scripture_guidance_when_unsupported_target(monkeypatch: pytest.MonkeyPatch):
    # Arrange: Italian unsupported; expect guidance message
    query = "translate gen 1:1-3 into italian"
    monkeypatch.setattr(brain_nodes.open_ai_client.responses, "parse", _make_parse_stub(query))
    state = _state_for(query)

    # Act
    out = brain_nodes.handle_translate_scripture(state)

    # Assert
    items = out.get("responses") or []
    msg = cast(str, items[0]["response"])  # guidance returns a string
    assert "Translating into Italian is currently not supported" in msg
