"""Helper functions for translation helps processing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, cast

from openai import OpenAI
from openai.types.responses.easy_input_message_param import EasyInputMessageParam

from bt_servant_engine.core.config import config
from bt_servant_engine.core.logging import get_logger
from bt_servant_engine.services.passage_selection import (
    resolve_selection_for_single_book,
)
from utils.bsb import (
    BOOK_MAP as BSB_BOOK_MAP,
    clamp_ranges_by_verse_limit,
    label_ranges,
    select_verses,
)
from utils.translation_helps import get_missing_th_books, select_translation_helps

logger = get_logger(__name__)

# Type alias for translation ranges
TranslationRange = tuple[int, int | None, int | None, int | None]


def _compact_translation_help_entries(entries: list[dict]) -> list[dict]:
    """Reduce translation help entries to essentials for the LLM payload."""
    compact: list[dict] = []
    for entry in entries:
        verse_text = cast(str, entry.get("ult_verse_text") or "")
        notes: list[dict] = []
        for note in cast(list[dict], entry.get("notes") or []):
            note_text = cast(str, note.get("note") or "")
            if not note_text:
                continue
            compact_note: dict[str, str] = {"note": note_text}
            quote = cast(Optional[str], note.get("orig_language_quote"))
            if quote:
                compact_note["orig_language_quote"] = quote
            notes.append(compact_note)
        compact.append(
            {
                "reference": entry.get("reference"),
                "verse_text": verse_text,
                "notes": notes,
            }
        )
    return compact


def prepare_translation_helps(
    client: OpenAI,
    query: str,
    query_lang: str,
    th_root: Path,
    bsb_root: Path,
    *,
    book_map: dict[str, Any],
    detect_mentioned_books_fn: callable,
    translate_text_fn: callable,
    selection_focus_hint: str | None = None,
) -> tuple[Optional[str], Optional[list[TranslationRange]], Optional[list[dict]], Optional[str]]:
    """Resolve canonical selection, enforce limits, and load raw help entries."""
    canonical_book, ranges, err = resolve_selection_for_single_book(
        client,
        query,
        query_lang,
        book_map,
        detect_mentioned_books_fn,
        translate_text_fn,
        focus_hint=selection_focus_hint,
    )
    if err:
        return None, None, None, err
    assert canonical_book is not None and ranges is not None

    missing_books = set(get_missing_th_books(th_root))
    if canonical_book in missing_books:
        return (
            None,
            None,
            None,
            (
                "Translation helps for "
                f"{BSB_BOOK_MAP[canonical_book]['ref_abbr']} are not available yet. "
                "Currently missing books: "
                f"{', '.join(sorted(BSB_BOOK_MAP[b]['ref_abbr'] for b in missing_books))}. "
                "Would you like translation help for one of the supported books instead?"
            ),
        )

    verse_count = len(select_verses(bsb_root, canonical_book, ranges))
    if verse_count > config.TRANSLATION_HELPS_VERSE_LIMIT:
        return (
            None,
            None,
            None,
            (
                "I can only provide translate help for "
                f"{config.TRANSLATION_HELPS_VERSE_LIMIT} verses at a time. "
                "Your selection "
                f"{label_ranges(canonical_book, ranges)} includes {verse_count} verses. "
                "Please narrow the range (e.g., a chapter or a shorter span)."
            ),
        )

    limited_ranges = clamp_ranges_by_verse_limit(
        bsb_root,
        canonical_book,
        ranges,
        max_verses=config.TRANSLATION_HELPS_VERSE_LIMIT,
    )
    if not limited_ranges:
        return (
            None,
            None,
            None,
            "I couldn't identify verses for that selection in the BSB index. Please try another reference.",
        )

    raw_helps = select_translation_helps(th_root, canonical_book, limited_ranges)
    logger.info("[translation-helps] selected %d help entries", len(raw_helps))
    if not raw_helps:
        return (
            None,
            None,
            None,
            "I couldn't locate translation helps for that selection. Please check the reference and try again.",
        )
    return canonical_book, list(limited_ranges), raw_helps, None


def build_translation_helps_context(
    canonical_book: str,
    ranges: list[TranslationRange],
    raw_helps: list[dict],
) -> tuple[str, dict[str, Any]]:
    """Return the reference label and compact JSON context for the LLM."""
    ref_label = label_ranges(canonical_book, ranges)
    context_obj = {
        "reference_label": ref_label,
        "selection": {
            "book": canonical_book,
            "ranges": [
                {
                    "start_chapter": sc,
                    "start_verse": sv,
                    "end_chapter": ec,
                    "end_verse": ev,
                }
                for (sc, sv, ec, ev) in ranges
            ],
        },
        "translation_helps": _compact_translation_help_entries(raw_helps),
    }
    return ref_label, context_obj


def build_translation_helps_messages(
    ref_label: str, context_obj: dict[str, object]
) -> list[EasyInputMessageParam]:
    """Construct the LLM messages for the translation helps prompt."""
    payload = json.dumps(context_obj, ensure_ascii=False)
    return [
        {
            "role": "developer",
            "content": "Focus only on the portion of the user's message that asked for translation helps. Ignore any other requests or book references in the message.",
        },
        {"role": "developer", "content": f"Selection: {ref_label}"},
        {"role": "developer", "content": "Use the JSON context below strictly:"},
        {"role": "developer", "content": payload},
        {
            "role": "user",
            "content": (
                "Using the provided context, explain the translation challenges and give actionable guidance for this selection."
            ),
        },
    ]


__all__ = [
    "TranslationRange",
    "prepare_translation_helps",
    "build_translation_helps_context",
    "build_translation_helps_messages",
]
