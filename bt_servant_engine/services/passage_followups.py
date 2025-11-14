"""Deterministic helpers for building passage follow-up suggestions."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, Iterable, Mapping, MutableMapping, Optional, Sequence, Tuple, cast

from bt_servant_engine.core.config import config
from bt_servant_engine.core.intents import IntentType
from bt_servant_engine.core.language import friendly_language_name
from bt_servant_engine.core.logging import get_logger
from utils.bsb import (
    BOOK_MAP,
    FULL_BOOK_SENTINEL,
    label_ranges,
    load_book_json,
    parse_ch_verse_from_reference,
)

logger = get_logger(__name__)

BSB_DATA_ROOT = Path("sources") / "bible_data" / "en" / "bsb"
BOOK_SEQUENCE = list(BOOK_MAP.keys())

PassageRange = Tuple[int, int, int]
RawRange = Tuple[int, int | None, int | None, int | None]
RANGE_TUPLE_LENGTH = 4


@dataclass(slots=True)
class _FollowupPosition:
    book: str
    chapter: int
    verse: int
    chapter_lengths: MutableMapping[int, int]


@lru_cache(maxsize=128)
def _chapter_lengths(book: str) -> MutableMapping[int, int]:
    """Return a mapping of chapter -> verse count for the given book."""
    mapping = BOOK_MAP.get(book)
    if not mapping:
        raise KeyError(f"Unknown canonical book '{book}'")

    entries = load_book_json(BSB_DATA_ROOT, mapping["file_stem"])
    lengths: dict[int, int] = {}
    for entry in entries:
        ref = entry.get("reference")
        if not isinstance(ref, str):
            continue
        parsed = parse_ch_verse_from_reference(ref)
        if parsed is None:
            continue
        chapter, verse = parsed
        current = lengths.get(chapter, 0)
        if verse > current:
            lengths[chapter] = verse
    return lengths


def _resolve_range_endpoint(
    range_tuple: RawRange,
    lengths: Mapping[int, int],
) -> tuple[int, int]:
    """Return the final (chapter, verse) that was delivered for a range tuple."""
    _start_chapter, _start_verse, end_chapter, end_verse = range_tuple
    if end_chapter is None or end_chapter >= FULL_BOOK_SENTINEL:
        end_chapter = max(lengths.keys())
    if end_verse is None or end_verse >= FULL_BOOK_SENTINEL:
        end_verse = lengths[end_chapter]
    return end_chapter, end_verse


def _next_book_name(book: str) -> str:
    """Return the canonical book that follows the provided book (wrap to Genesis)."""
    try:
        idx = BOOK_SEQUENCE.index(book)
    except ValueError as exc:
        raise KeyError(f"Unknown canonical book '{book}'") from exc
    return BOOK_SEQUENCE[(idx + 1) % len(BOOK_SEQUENCE)]


def _normalize_ranges(ranges: Sequence[Iterable[int | None]]) -> list[RawRange]:
    """Return ranges as concrete tuples; coerce inputs like lists to tuples."""
    out: list[RawRange] = []
    for r in ranges:
        items = list(r)
        if len(items) != RANGE_TUPLE_LENGTH:
            raise ValueError(f"Expected 4-tuple range, got {items!r}")
        start_chapter = items[0]
        if start_chapter is None:
            raise ValueError("start_chapter cannot be None for normalized ranges")
        out.append(
            (
                cast(int, start_chapter),
                None if items[1] is None else cast(int, items[1]),
                None if items[2] is None else cast(int, items[2]),
                None if items[3] is None else cast(int, items[3]),
            )
        )
    return out


class _PassageFollowupAbort(Exception):
    """Internal control-flow exception for deterministic follow-up selection."""


def _validate_ranges(ranges: Sequence[Iterable[int | None]]) -> list[RawRange]:
    if not ranges:
        logger.warning("[passage-followup] Empty ranges supplied; skipping suggestion")
        raise _PassageFollowupAbort
    try:
        return _normalize_ranges(ranges)
    except ValueError as exc:  # pragma: no cover - defensive guard
        logger.exception("[passage-followup] Invalid ranges supplied; skipping suggestion")
        raise _PassageFollowupAbort from exc


def _prepare_initial_position(
    book: str, ranges: Sequence[Iterable[int | None]]
) -> _FollowupPosition:
    normalized = _validate_ranges(ranges)
    return _start_followup_position(book, normalized)


def _start_followup_position(book: str, normalized_ranges: Sequence[RawRange]) -> _FollowupPosition:
    try:
        chapter_lengths = _chapter_lengths(book)
    except KeyError as exc:
        logger.exception("[passage-followup] Unknown book '%s'; skipping suggestion", book)
        raise _PassageFollowupAbort from exc

    last_range = normalized_ranges[-1]
    end_chapter, end_verse = _resolve_range_endpoint(last_range, chapter_lengths)
    return _FollowupPosition(
        book=book,
        chapter=end_chapter,
        verse=end_verse + 1,
        chapter_lengths=chapter_lengths,
    )


def _advance_position(position: _FollowupPosition) -> _FollowupPosition:
    chapter_length = position.chapter_lengths.get(position.chapter, 0)
    if 0 < chapter_length >= position.verse:
        return position

    next_book = position.book
    next_chapter = position.chapter + 1
    chapter_lengths = position.chapter_lengths

    if next_chapter not in chapter_lengths:
        next_book = _next_book_name(position.book)
        try:
            chapter_lengths = _chapter_lengths(next_book)
        except KeyError as exc:  # pragma: no cover - defensive guard
            logger.exception(
                "[passage-followup] Unknown next book '%s'; skipping suggestion", next_book
            )
            raise _PassageFollowupAbort from exc
        next_chapter = 1

    return _FollowupPosition(
        book=next_book,
        chapter=next_chapter,
        verse=1,
        chapter_lengths=chapter_lengths,
    )


def _chapter_max(position: _FollowupPosition) -> int:
    chapter_length = position.chapter_lengths.get(position.chapter, 0)
    if chapter_length <= 0:
        logger.warning(
            "[passage-followup] No verse data recorded for %s %s; skipping suggestion",
            position.book,
            position.chapter,
        )
        raise _PassageFollowupAbort
    return chapter_length


def _finalize_position(book: str, ranges: Sequence[Iterable[int | None]]) -> _FollowupPosition:
    position = _prepare_initial_position(book, ranges)
    return _advance_position(position)


def _build_suggestion(position: _FollowupPosition, verse_limit: int) -> tuple[str, PassageRange]:
    chapter_max = _chapter_max(position)
    end_suggested = min(position.verse + verse_limit - 1, chapter_max)
    return position.book, (position.chapter, position.verse, end_suggested)


def _propose_followup_impl(
    book: str,
    ranges: Sequence[Iterable[int | None]],
    verse_limit: int,
) -> Optional[tuple[str, PassageRange]]:
    try:
        return _build_suggestion(_finalize_position(book, ranges), verse_limit)
    except _PassageFollowupAbort:
        return None


def propose_next_passage_range(
    book: str,
    ranges: Sequence[Iterable[int | None]],
    verse_limit: int,
) -> Optional[tuple[str, PassageRange]]:
    """Return the next single-chapter range to propose after the provided selection.

    The algorithm walks forward from the delivered verses by up to ``verse_limit``
    verses. Ranges never cross chapters or books. When the end of a book is reached,
    the suggestion wraps to the first chapter of the next canonical book. After
    Revelation, the suggestion wraps back to Genesis.
    """
    if verse_limit <= 0:
        logger.warning("[passage-followup] verse_limit <= 0; skipping suggestion")
        return None

    return _propose_followup_impl(book, ranges, verse_limit)


ENGLISH_FOLLOWUP_TEMPLATES: dict[IntentType, str] = {
    IntentType.GET_PASSAGE_SUMMARY: "Would you like me to summarize {label} next?",
    IntentType.GET_PASSAGE_KEYWORDS: "Would you like me to show key terms from {label}?",
    IntentType.GET_TRANSLATION_HELPS: "Would you like translation helps for {label}?",
    IntentType.RETRIEVE_SCRIPTURE: "Would you like me to retrieve {label}?",
    IntentType.LISTEN_TO_SCRIPTURE: "Would you like me to read {label} aloud?",
    IntentType.TRANSLATE_SCRIPTURE: "Would you like me to translate {label}?",
}


def _suggestion_label(suggestion: tuple[str, PassageRange]) -> str:
    next_book, (chapter, start_verse, end_verse) = suggestion
    return label_ranges(next_book, [(chapter, start_verse, chapter, end_verse)])


def build_followup_question(
    intent: IntentType,
    context: Mapping[str, object],
    target_language: Optional[str],
    translate_text_fn: Optional[Callable[[str, str], str]] = None,
) -> Optional[str]:
    """Return a localized deterministic follow-up question for passage intents."""
    template = ENGLISH_FOLLOWUP_TEMPLATES.get(intent)
    if not template:
        return None

    book = context.get("book")
    ranges = context.get("ranges")
    if not isinstance(book, str) or not isinstance(ranges, Sequence):
        logger.debug("[passage-followup] Missing or invalid context for intent=%s", intent.value)
        return None

    suggestion = propose_next_passage_range(
        book,
        cast(Sequence[Iterable[int | None]], ranges),
        config.TRANSLATION_HELPS_VERSE_LIMIT,
    )
    if not suggestion:
        return None

    label = _suggestion_label(suggestion)
    target_label = None
    if intent is IntentType.TRANSLATE_SCRIPTURE:
        raw_target = context.get("target_language")
        if isinstance(raw_target, str) and raw_target.strip():
            target_code = raw_target.strip().lower()
            target_label = friendly_language_name(target_code, fallback=raw_target.strip().title())

    if target_label:
        english_question = f"Would you like me to translate {label} into {target_label}?"
    else:
        english_question = template.format(label=label)

    lang = (target_language or "").strip().lower() or "en"
    if lang == "en" or translate_text_fn is None:
        return english_question

    try:
        return translate_text_fn(english_question, lang)
    except Exception:  # pylint: disable=broad-except
        logger.exception(
            "[passage-followup] Failed to translate follow-up to language '%s'; using English",
            lang,
        )
        return english_question


__all__ = [
    "build_followup_question",
    "propose_next_passage_range",
]
