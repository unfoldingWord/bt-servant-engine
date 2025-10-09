"""Tests for listen-to-scripture delegating to retrieve-scripture and setting voice flag."""
# pylint: disable=missing-module-docstring,too-few-public-methods,missing-function-docstring,unused-argument,duplicate-code

import re
from typing import Any, cast

import pytest

from bt_servant_engine.core.language import Language, ResponseLanguage
from bt_servant_engine.core.models import PassageRef, PassageSelection
from bt_servant_engine.services import brain_nodes
from bt_servant_engine.services.brain_orchestrator import BrainState


class _StubParseResult:
    def __init__(self, output_parsed: Any):
        self.output_parsed = output_parsed
        self.usage = None


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


def test_listen_to_scripture_sets_voice_and_formats_paragraph(monkeypatch: pytest.MonkeyPatch):
    # Arrange: stub selection parse; no explicit requested language
    def parse_stub(*args: Any, **kwargs: Any):  # noqa: ANN401 - test stub
        tf = kwargs.get("text_format")
        if tf is PassageSelection:
            sel = PassageSelection(
                selections=[
                    PassageRef(
                        book="Genesis",
                        start_chapter=1,
                        start_verse=1,
                        end_chapter=1,
                        end_verse=3,
                    )
                ]
            )
            return _StubParseResult(sel)
        if tf is ResponseLanguage:
            # No explicit requested language in message
            return _StubParseResult(ResponseLanguage(language=Language.OTHER))
        return _StubParseResult(None)

    monkeypatch.setattr(brain_nodes.open_ai_client.responses, "parse", parse_stub)

    state = _state_for("Please read Genesis 1:1-3 out loud")

    # Act
    out = brain_nodes.handle_listen_to_scripture(state)

    # Assert: voice flag present
    assert out.get("send_voice_message") is True
    # Structured scripture response with a flowing paragraph
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
