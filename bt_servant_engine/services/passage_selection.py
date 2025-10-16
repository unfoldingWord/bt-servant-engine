"""Passage selection parsing and normalization for Bible references."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Optional, cast

from openai import OpenAI
from openai.types.responses.easy_input_message_param import EasyInputMessageParam

from bt_servant_engine.core.language import Language
from bt_servant_engine.core.logging import get_logger
from bt_servant_engine.core.models import PassageRef, PassageSelection
from bt_servant_engine.services.openai_utils import extract_cached_input_tokens, track_openai_usage
from utils.bsb import FULL_BOOK_SENTINEL, normalize_book_name
from utils.perf import add_tokens

logger = get_logger(__name__)

_MULTI_BOOK_MENTION_THRESHOLD = 2
_SINGLE_BOOK_MENTION_COUNT = 1

PASSAGE_SELECTION_AGENT_SYSTEM_PROMPT = """
# Identity

You classify the user's message to extract explicit Bible passage references.
Return a normalized, structured selection of book + verse ranges.

# Instructions

- Obey any developer instructions that precede the user message. They may narrow the
  focus to a specific clause of the user's text. When instructed, ignore unrelated
  books or requests outside that clause.
- Only choose from these canonical book names (exact match):
  {books}
- Accept a variety of phrasings (e.g., "John 3:16", "Jn 3:16–18", "1 John 2:1-3",
  "Psalm 1", "Song of Songs 2").
- Normalize all book names to the exact canonical name.
- Numbered books are distinct canonical names. When a leading number precedes a book
  (e.g., "1 John", "2 Samuel", "3 John"), treat the number as part of the book name —
  it is NOT a chapter reference. For example, "3 John" means the book "3 John"
  (and the whole book if no chapter/verse is given), whereas "John 3" means chapter 3 of "John".
- Support:
  - Single verse (John 3:16)
  - Verse ranges within a chapter (John 3:16-18)
  - Cross-chapter within a single book (John 3:16–4:2)
  - Whole chapters (John 3)
  - Multi-chapter ranges with no verse specification (e.g., "John 1–4",
    "John chapters 1–4"): set start_chapter=1, end_chapter=4 and leave verses empty
  - Whole book (John)
  - Multiple disjoint ranges within the same book (comma/semicolon separated)
- Do not cross books in one selection. If the user mentions multiple books (including
  with 'and', commas, or hyphens like 'Gen–Exo') and a single book cannot be
  unambiguously inferred by explicit chapter/verse qualifiers, return an empty
  selection. Prefer a clearly qualified single book (e.g., "Mark 1:1") over earlier
  mentions without qualifiers.
- If no verses/chapters are supplied for the chosen book, interpret it as the whole book.
- If no clear passage is present (and no book can be reasonably inferred), return an empty list.

# Output format
Return JSON parsable into the provided schema.

# Examples
- "What are the keywords in Genesis and Exodus?" -> return empty selection (multiple
  books; no clear single-book qualifier).
- "Gen–Exo" -> return empty selection (multiple books; no clear single-book qualifier).
- "John and Mark 1:1" -> choose Mark 1:1 (explicit qualifier picks Mark over first mention).
- "summarize 3 John" -> choose book "3 John" with no chapters/verses (whole book selection).
- "summarize John 3" -> choose book "John" with start_chapter=3 (whole chapter if no verses).
- Developer hint: "Focus only on the portion asking for translation helps."
  User message: "I want to listen to John 1:1, and I also want help translating Gal 1:3-4."
  -> choose Galatians 1:3-4 (ignore the listening request entirely).
"""


@dataclass(slots=True)
class PassageSelectionDependencies:
    """External services required for passage selection parsing."""

    client: OpenAI
    book_map: dict[str, Any]
    detect_mentioned_books: Callable[[str], list[str]]
    translate_text: Callable[..., str]


@dataclass(slots=True)
class PassageSelectionRequest:
    """Structured request describing the user query and parsing dependencies."""

    query: str
    query_lang: str
    dependencies: PassageSelectionDependencies
    focus_hint: Optional[str] = None


def _prepare_parse_input(request: PassageSelectionRequest) -> str:
    if request.query_lang == Language.ENGLISH.value:
        logger.info("[selection-helper] parsing in English (no translation needed)")
        return request.query
    logger.info("[selection-helper] translating query to English for parsing")
    return request.dependencies.translate_text(request.query, "en")


def _build_selection_messages(
    parse_input: str, request: PassageSelectionRequest
) -> list[EasyInputMessageParam]:
    books = ", ".join(request.dependencies.book_map.keys())
    system_prompt = PASSAGE_SELECTION_AGENT_SYSTEM_PROMPT.format(books=books)
    messages: list[EasyInputMessageParam] = [
        cast(EasyInputMessageParam, {"role": "system", "content": system_prompt})
    ]
    if request.focus_hint:
        logger.info("[selection-helper] applying focus hint: %s", request.focus_hint)
        messages.append(
            cast(EasyInputMessageParam, {"role": "developer", "content": request.focus_hint})
        )
    messages.append(cast(EasyInputMessageParam, {"role": "user", "content": parse_input}))
    return messages


def _run_selection_model(
    messages: list[EasyInputMessageParam], dependencies: PassageSelectionDependencies
) -> PassageSelection:
    logger.info("[selection-helper] extracting passage selection via LLM")
    selection_resp = dependencies.client.responses.parse(
        model="gpt-4o",
        input=cast(Any, messages),
        text_format=PassageSelection,
        store=False,
    )
    usage = getattr(selection_resp, "usage", None)
    track_openai_usage(usage, "gpt-4o", extract_cached_input_tokens, add_tokens)
    return cast(PassageSelection, selection_resp.output_parsed)


def _apply_chapter_range_heuristic(selection: PassageSelection, parse_input: str) -> None:
    lower_in = parse_input.lower()
    chap_match = re.search(r"\bchapters?\s+(\d+)\s*[-–]\s*(\d+)\b", lower_in)
    if chap_match and selection.selections:
        a, b = int(chap_match.group(1)), int(chap_match.group(2))
        logger.info(
            (
                "[selection-helper] correcting to multi-chapter range: %d-%d due to "
                "'chapters' phrasing"
            ),
            a,
            b,
        )
        first = selection.selections[0]
        selection.selections[0] = PassageRef(
            book=first.book,
            start_chapter=a,
            start_verse=None,
            end_chapter=b,
            end_verse=None,
        )


def _empty_selection_response(
    mentioned_books: list[str],
) -> tuple[str | None, list[tuple[int, int | None, int | None, int | None]] | None, str | None]:
    if len(mentioned_books) >= _MULTI_BOOK_MENTION_THRESHOLD:
        msg = (
            "Please request a selection for one book at a time. "
            "If you need multiple books, send a separate message for each."
        )
        logger.info("[selection-helper] empty parse; multiple books detected -> cross-book message")
        return None, None, msg
    if len(mentioned_books) == _SINGLE_BOOK_MENTION_COUNT:
        primary = mentioned_books[0]
        logger.info(
            "[selection-helper] empty parse; falling back to single detected book: %s", primary
        )
        return primary, [(1, None, FULL_BOOK_SENTINEL, None)], None
    msg = (
        "I couldn't identify a clear Bible passage in your request. Supported "
        "selection types include: single verse (e.g., John 3:16); verse range "
        "within a chapter (John 3:16-18); cross-chapter within a single book "
        "(John 3:16–4:2); whole chapter (John 3); multi-chapter span with no "
        "verses (John 1–4); or the whole book (John). Multiple books in one "
        "request are not supported — please choose one book."
    )
    logger.info("[selection-helper] no passage detected; returning guidance message")
    return None, None, msg


def _normalize_selection(
    selection: PassageSelection, book_map: dict[str, Any]
) -> tuple[str | None, list[PassageRef] | None, str | None]:
    canonical_books: list[str] = []
    normalized_selections: list[PassageRef] = []
    for sel in selection.selections:
        canonical = normalize_book_name(sel.book) or sel.book
        if canonical not in book_map:
            supported = ", ".join(book_map.keys())
            msg = (
                f"The book '{sel.book}' is not recognized. Please use a supported "
                f"canonical book name. Supported books include: {supported}."
            )
            logger.info("[selection-helper] unsupported book requested: %s", sel.book)
            return None, None, msg
        canonical_books.append(canonical)
        normalized_selections.append(
            PassageRef(
                book=canonical,
                start_chapter=sel.start_chapter,
                start_verse=sel.start_verse,
                end_chapter=sel.end_chapter,
                end_verse=sel.end_verse,
            )
        )
    if len(set(canonical_books)) != 1:
        msg = (
            "Please request a selection for one book at a time. "
            "If you need multiple books, send a separate message for each."
        )
        logger.info("[selection-helper] cross-book selection detected")
        return None, None, msg
    return canonical_books[0], normalized_selections, None


def _ranges_from_selections(
    selections: list[PassageRef],
) -> list[tuple[int, int | None, int | None, int | None]]:
    ranges: list[tuple[int, int | None, int | None, int | None]] = []
    for sel in selections:
        if sel.start_chapter is None:
            ranges.append((1, None, FULL_BOOK_SENTINEL, None))
        else:
            ranges.append((sel.start_chapter, sel.start_verse, sel.end_chapter, sel.end_verse))
    return ranges


def resolve_selection_for_single_book(
    request: PassageSelectionRequest,
) -> tuple[str | None, list[tuple[int, int | None, int | None, int | None]] | None, str | None]:
    """Parse and normalize a user query into a single canonical book and ranges.

    Args:
        request: Structured selection request containing query metadata and dependencies.

    Returns:
        Tuple of (canonical_book, ranges, error_message). On success, the
        error_message is None. On failure, canonical_book and ranges are None and
        error_message contains a user-friendly explanation.

        If ``focus_hint`` is provided, it is sent as a developer message to steer the
        selection model toward the clause relevant to the current intent.
    """
    logger.info(
        "[selection-helper] start; query_lang=%s; query=%s",
        request.query_lang,
        request.query,
    )
    parse_input = _prepare_parse_input(request)
    messages = _build_selection_messages(parse_input, request)
    selection = _run_selection_model(messages, request.dependencies)
    logger.info("[selection-helper] extracted %d selection(s)", len(selection.selections))

    mentioned = request.dependencies.detect_mentioned_books(parse_input)
    _apply_chapter_range_heuristic(selection, parse_input)

    if not selection.selections:
        return _empty_selection_response(mentioned)

    canonical_book, normalized_selections, error = _normalize_selection(
        selection, request.dependencies.book_map
    )
    if error:
        return None, None, error
    assert canonical_book is not None  # nosec B101
    assert normalized_selections is not None  # nosec B101
    ranges = _ranges_from_selections(normalized_selections)
    logger.info("[selection-helper] canonical_book=%s", canonical_book)
    logger.info("[selection-helper] ranges=%s", ranges)
    return canonical_book, ranges, None


__all__ = [
    "PASSAGE_SELECTION_AGENT_SYSTEM_PROMPT",
    "PassageSelectionDependencies",
    "PassageSelectionRequest",
    "resolve_selection_for_single_book",
]
