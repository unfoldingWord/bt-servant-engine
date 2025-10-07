"""Focused unit tests for intent service helpers."""

# pylint: disable=missing-class-docstring,missing-function-docstring,too-few-public-methods

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, List, Sequence, cast

import pytest

from bt_servant_engine.core.intents import IntentType
from bt_servant_engine.services.intents import (
    consult_fia_resources as fia_module,
    retrieve_scripture as rs_module,
    translation_helps as th_module,
)


class DummyOpenAI:
    """Minimal OpenAI stub exposing a responses namespace."""

    def __init__(self, response_text: str = "", raise_on_call: bool = False) -> None:
        self.calls: list[dict[str, Any]] = []
        self.response_text = response_text
        self.raise_on_call = raise_on_call
        self.responses = self._Responses(self)

    class _Responses:
        def __init__(self, outer: "DummyOpenAI") -> None:
            self._outer = outer

        def create(self, **kwargs: Any) -> SimpleNamespace:
            if self._outer.raise_on_call:
                raise AssertionError("OpenAI.create should not have been called")
            self._outer.calls.append(kwargs)
            return SimpleNamespace(output_text=self._outer.response_text, usage=None)

        def parse(self, **kwargs: Any) -> SimpleNamespace:
            if self._outer.raise_on_call:
                raise AssertionError("OpenAI.parse should not have been called")
            self._outer.calls.append(kwargs)
            return SimpleNamespace(output_parsed=None, usage=None)


class DummyChroma:
    """Simple Chroma stub that yields canned responses."""

    def __init__(self, responses: dict[str, Any] | None = None) -> None:
        self.responses = responses or {}
        self.calls: list[str] = []

    def query_collection(self, name: str, **_: Any) -> Any:
        self.calls.append(name)
        result = self.responses.get(name)
        if isinstance(result, Exception):
            raise result
        return result


def test_retrieve_scripture_returns_structured_passage(monkeypatch: pytest.MonkeyPatch) -> None:
    """Happy-path retrieval returns structured payload without triggering translation."""

    dummy_ai = DummyOpenAI()
    monkeypatch.setattr(rs_module, "_detect_requested_language", lambda **_: None)

    def _static_resolver(*_args: Any, **_kwargs: Any) -> tuple[Path, str, str]:
        return (Path("/tmp"), "en", "bsb")

    monkeypatch.setattr(rs_module, "resolve_bible_data_root", _static_resolver)
    monkeypatch.setattr(
        rs_module,
        "select_verses",
        lambda *_args, **_kwargs: [("ref1", "Verse one"), ("ref2", "Verse two")],
    )
    monkeypatch.setattr(rs_module, "label_ranges", lambda *_args, **_kwargs: "John 1:1-2")
    monkeypatch.setattr(rs_module, "load_book_titles", lambda *_args, **_kwargs: {"John": "John"})
    monkeypatch.setattr(rs_module, "get_book_name", lambda *_args, **_kwargs: "John")

    def passthrough_translate(text: str, code: str, strength: str | None) -> str:
        return f"{code}:{text}:{strength}"

    result = rs_module.handle_retrieve_scripture_intent(
        openai_client=cast(Any, dummy_ai),
        user_query="Read John 1:1-2",
        canonical_book="John",
        ranges=[(1, 1, 1, 2)],
        agentic_strength="normal",
        user_response_language=None,
        query_language="en",
        target_language_prompt="prompt",
        translate_text_func=passthrough_translate,
    )

    assert result["responses"][0]["intent"] is IntentType.RETRIEVE_SCRIPTURE
    payload = result["responses"][0]["response"]
    assert payload["content_language"] == "en"
    assert payload["header_is_translated"] is False
    assert payload["segments"][0]["text"] == "John"
    assert payload["segments"][2]["text"] == "Verse one Verse two"
    # Ensure auto-translation path was skipped.
    assert not dummy_ai.calls, "language detection should be bypassed via stub"


def test_retrieve_scripture_translates_when_target_requested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Auto-translation path invokes translate helper and returns target language content."""

    monkeypatch.setattr(rs_module, "_detect_requested_language", lambda **_: "es")

    call_counter = {"count": 0}

    def fake_resolver(**kwargs: Any) -> tuple[Path, str, str]:
        call_counter["count"] += 1
        requested = kwargs.get("requested_lang")
        if call_counter["count"] == 1:
            return Path("/tmp"), "en", "base-version"
        lang = (
            requested
            or kwargs.get("response_language")
            or kwargs.get("query_language")
            or "es"
        )
        return Path("/tmp"), str(lang), "target-version"

    monkeypatch.setattr(rs_module, "resolve_bible_data_root", fake_resolver)
    monkeypatch.setattr(
        rs_module,
        "select_verses",
        lambda *_args, **_kwargs: [("ref1", "Verse one"), ("ref2", "Verse two")],
    )
    monkeypatch.setattr(rs_module, "label_ranges", lambda *_args, **_kwargs: "John 1:1-2")
    monkeypatch.setattr(rs_module, "load_book_titles", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(rs_module, "get_book_name", lambda *_args, **_kwargs: "John")

    translations: list[tuple[str, str, str | None]] = []

    def translate(text: str, code: str, strength: str | None) -> str:
        translations.append((text, code, strength))
        return f"{code}:{text}"

    result = rs_module.handle_retrieve_scripture_intent(
        openai_client=cast(Any, DummyOpenAI()),
        user_query="Read John 1:1-2 in Spanish",
        canonical_book="John",
        ranges=[(1, 1, 1, 2)],
        agentic_strength="normal",
        user_response_language=None,
        query_language="en",
        target_language_prompt="prompt",
        translate_text_func=translate,
    )

    payload = result["responses"][0]["response"]
    assert payload["content_language"] == "es"
    assert payload["header_is_translated"] is True
    assert payload["segments"][0]["text"] == "es:John"
    assert payload["segments"][2]["text"] == "es:Verse one es:Verse two"
    # One call for header + one per verse.
    assert len(translations) == 3
    assert all(call[1] == "es" for call in translations)


def test_translation_helps_missing_book_returns_message(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unavailable book should yield a helpful user message."""

    monkeypatch.setattr(th_module, "get_missing_th_books", lambda *_args, **_kwargs: ["John"])

    out = th_module.handle_translation_helps_intent(
        openai_client=cast(Any, DummyOpenAI(raise_on_call=True)),
        canonical_book="John",
        ranges=[(1, 1, 1, 2)],
        model_name="gpt-test",
    )

    response = out["responses"][0]
    assert response["intent"] is IntentType.GET_TRANSLATION_HELPS
    assert "not available" in response["response"]


def test_translation_helps_generates_response(monkeypatch: pytest.MonkeyPatch) -> None:
    """Happy-path translation helps call invokes OpenAI and returns prose."""

    monkeypatch.setattr(th_module, "get_missing_th_books", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(th_module, "select_verses", lambda *_args, **_kwargs: [("ref", "1:1")])

    def fake_clamp(
        _bsb_root: Path,
        _book: str,
        ranges_list: Sequence[Any],
        *,
        max_verses: int,
    ) -> List[Any]:
        del max_verses
        return list(ranges_list)

    monkeypatch.setattr(th_module, "clamp_ranges_by_verse_limit", fake_clamp)
    fake_entries = [
        {
            "reference": "John 1:1",
            "notes": [{"note": "Focus on meaning."}],
            "ult_verse_text": "Verse",
        }
    ]
    monkeypatch.setattr(
        th_module,
        "select_translation_helps",
        lambda *_args, **_kwargs: fake_entries,
    )
    monkeypatch.setattr(th_module, "label_ranges", lambda *_args, **_kwargs: "John 1:1")

    dummy_ai = DummyOpenAI(response_text="Guidance text")

    out = th_module.handle_translation_helps_intent(
        openai_client=cast(Any, dummy_ai),
        canonical_book="John",
        ranges=[(1, 1, 1, 1)],
        model_name="gpt-test",
    )

    response = out["responses"][0]
    assert response["intent"] is IntentType.GET_TRANSLATION_HELPS
    assert response["response"].startswith("Translation helps for John 1:1")
    assert dummy_ai.calls, "OpenAI should be invoked for translation helps content"


def test_consult_fia_returns_fallback_when_no_context(monkeypatch: pytest.MonkeyPatch) -> None:
    """No vector docs and no reference content should return fallback guidance."""

    _ = monkeypatch
    chroma = DummyChroma(responses={"fr_fia_resources": []})
    out = fia_module.handle_consult_fia_resources_intent(
        openai_client=cast(Any, DummyOpenAI(raise_on_call=True)),
        user_query="How do I proceed?",
        chat_history=[],
        user_response_language="fr",
        query_language=None,
        model_name="gpt-test",
        boilerplate_message="Please reach out for help.",
        chroma=cast(Any, chroma),
        fia_reference_content="",
    )

    response = out["responses"][0]
    assert response["intent"] is IntentType.CONSULT_FIA_RESOURCES
    assert "couldn't find any FIA resources" in response["response"]


def test_consult_fia_generates_guidance(monkeypatch: pytest.MonkeyPatch) -> None:
    """Happy-path FIA consult should aggregate context and return model output."""

    _ = monkeypatch
    chroma = DummyChroma(
        responses={
            "fr_fia_resources": {
                "documents": [["Relevant passage"]],
                "distances": [[0.1]],
                "metadatas": [[{"name": "Doc", "source": "fia/fr.md"}]],
            }
        }
    )
    dummy_ai = DummyOpenAI(response_text="FIA guidance")

    out = fia_module.handle_consult_fia_resources_intent(
        openai_client=cast(Any, dummy_ai),
        user_query="What does FIA say about planning?",
        chat_history=[{"role": "user", "content": "Hello"}],
        user_response_language="fr",
        query_language=None,
        model_name="gpt-test",
        boilerplate_message="fallback",
        chroma=cast(Any, chroma),
        fia_reference_content="Reference content",
    )

    response = out["responses"][0]
    assert response["intent"] is IntentType.CONSULT_FIA_RESOURCES
    assert response["response"] == "FIA guidance"
    assert out["collection_used"] == "fr_fia_resources"
    assert dummy_ai.calls, "OpenAI should be invoked when context is available"
    assert chroma.calls == ["fr_fia_resources"]
