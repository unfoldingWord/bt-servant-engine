"""Unit tests for translation helpers."""

from __future__ import annotations

from typing import Any

from bt_servant_engine.services import translation_helpers as helpers

# pylint: disable=missing-function-docstring


def test_compact_translation_help_entries_filters_and_preserves() -> None:
    entries = [
        {
            "reference": "John 1:1",
            "ult_verse_text": "In the beginning...",
            "notes": [
                {"note": "Consider the tense", "orig_language_quote": "Ἐν ἀρχῇ"},
                {"note": ""},  # dropped
                {},  # dropped
            ],
        }
    ]
    compact = helpers._compact_translation_help_entries(entries)  # pylint: disable=protected-access
    assert compact == [
        {
            "reference": "John 1:1",
            "verse_text": "In the beginning...",
            "notes": [{"note": "Consider the tense", "orig_language_quote": "Ἐν ἀρχῇ"}],
        }
    ]


def test_build_translation_helps_context_shapes_payload() -> None:
    ranges: list[helpers.TranslationRange] = [(1, 1, 1, 2)]
    info: list[dict[str, Any]] = [
        {
            "reference": "John 1:1-2",
            "ult_verse_text": "In the beginning",
            "notes": [{"note": "Example"}],
        }
    ]
    ref_label, context = helpers.build_translation_helps_context("John", ranges, info)
    assert ref_label == "John 1:1-2"
    assert context["reference_label"] == "John 1:1-2"
    assert context["selection"]["book"] == "John"
    assert context["translation_helps"][0]["notes"][0]["note"] == "Example"


def test_build_translation_helps_context_includes_original_ranges() -> None:
    info: list[dict[str, Any]] = [
        {
            "reference": "Gen 1:1-5",
            "ult_verse_text": "In the beginning",
            "notes": [{"note": "Example"}],
        }
    ]
    ranges: list[helpers.TranslationRange] = [(1, 1, 1, 5)]
    original: list[helpers.TranslationRange] = [(1, 1, 1, 11)]
    _, context = helpers.build_translation_helps_context(
        "Genesis", ranges, info, original_ranges=original
    )
    selection = context["selection"]
    assert selection["original_ranges"][0]["end_verse"] == 11
    assert selection["original_reference_label"] == "Genesis 1:1-11"


def test_build_translation_helps_messages_includes_payload() -> None:
    ref_label = "John 1:1-2"
    context: dict[str, object] = {
        "reference_label": ref_label,
        "selection": {"book": "John", "ranges": []},
        "translation_helps": [],
    }
    messages = helpers.build_translation_helps_messages(
        ref_label, context, selection_note="Explain truncation"
    )
    assert len(messages) == 6
    assert messages[1]["role"] == "developer"
    assert "Explain truncation" in messages[1]["content"]
    assert ref_label in messages[2]["content"]
    assert messages[-1]["role"] == "user"
