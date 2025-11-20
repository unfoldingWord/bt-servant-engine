"""Unit tests for translate-scripture intent permutations."""
# pylint: disable=missing-module-docstring,line-too-long,too-few-public-methods,unused-argument,missing-function-docstring

from typing import Any, cast

import pytest

from bt_servant_engine.core.language import ResponseLanguage, TranslatedPassage
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
                return _StubParseResult(ResponseLanguage(language="nl"))
            if "chinese" in q:
                return _StubParseResult(ResponseLanguage(language="zh"))
            # For unsupported/ambiguous languages fall back to "other"
            return _StubParseResult(ResponseLanguage(language="other"))

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


@pytest.mark.parametrize(
    ("query", "target_code", "translated_body"),
    [
        (
            "translate gen 1:1 into turkish",
            "tr",
            "Başlangıçta...",
        ),
        (
            "translate gen 1:1 into amharic",
            "am",
            "Amharic translation...",
        ),
    ],
)
def test_translate_scripture_translates_with_arbitrary_target(
    monkeypatch: pytest.MonkeyPatch, query: str, target_code: str, translated_body: str
):
    # Arrange: target language was previously unsupported; ensure it now works

    def parse_stub(*args: Any, **kwargs: Any):  # noqa: ANN401 - test stub
        tf = kwargs.get("text_format")
        if tf is ResponseLanguage:
            return _StubParseResult(ResponseLanguage(language=target_code))
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
                body=translated_body,
                content_language=target_code,
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
    assert resp.get("content_language") == target_code
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


def test_translate_scripture_guidance_when_language_unspecified(monkeypatch: pytest.MonkeyPatch):
    # Arrange: Parser cannot infer target and no fallbacks available; expect guidance message
    query = "translate gen 1:1-3"
    monkeypatch.setattr(brain_nodes.open_ai_client.responses, "parse", _make_parse_stub(query))
    monkeypatch.setattr(
        brain_nodes,
        "translate_text",
        lambda response_text, target_language, *, agentic_strength=None: response_text,  # noqa: ANN001
    )
    state = _state_for(query)
    state["query_language"] = ""  # remove fallback so guidance path triggers

    # Act
    out = brain_nodes.handle_translate_scripture(state)

    # Assert
    items = out.get("responses") or []
    msg = cast(str, items[0]["response"])  # guidance returns a string
    assert "couldn't determine how to translate" in msg.lower()
