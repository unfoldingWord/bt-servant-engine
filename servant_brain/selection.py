# pylint: disable=line-too-long,too-many-arguments,too-many-locals,too-many-branches,too-many-statements,duplicate-code
"""Passage selection parsing helper.

Provides a resolver that extracts a single-book selection and normalizes it
to canonical book and verse ranges. The heavy lifting is delegated via
parameters to avoid imports back into the orchestration module.
"""

from __future__ import annotations

import re
from typing import Any, Callable, List, Type, cast

from openai import OpenAI
from openai.types.responses.easy_input_message_param import EasyInputMessageParam

from logger import get_logger

logger = get_logger(__name__)


def _book_patterns(books_map: dict[str, Any]) -> list[tuple[str, str]]:
    pats: list[tuple[str, str]] = []
    for canonical in books_map.keys():
        esc = re.escape(canonical)
        pats.append((canonical, rf"\b{esc}\b"))
    return pats


def _detect_mentioned_books(text: str, books_map: dict[str, Any]) -> list[str]:
    found: list[tuple[int, str]] = []
    lower = text
    for canonical, pattern in _book_patterns(books_map):
        for m in re.finditer(pattern, lower, flags=re.IGNORECASE):
            found.append((m.start(), canonical))
    found.sort(key=lambda t: t[0])
    seen = set()
    ordered: list[str] = []
    for _, can in found:
        if can not in seen:
            seen.add(can)
            ordered.append(can)
    return ordered


def resolve_selection_for_single_book(  # noqa: C901
    *,
    query: str,
    query_lang: str,
    open_ai_client: OpenAI,
    PassageSelection: Type[Any],
    PassageRef: Type[Any],
    add_tokens: Callable[..., None],
    extract_cached_input_tokens: Callable[[Any], int | None],
    translate_text: Callable[[str, str], str],
    books_map: dict[str, Any],
    normalize_book_name: Callable[[str], str | None],
    selection_prompt_template: str,
) -> tuple[str | None, list[tuple[int, int | None, int | None, int | None]] | None, str | None]:
    """Core resolver used by brain._resolve_selection_for_single_book.

    Returns (canonical_book, ranges, error_message).
    """
    logger.info("[selection-helper] start; query_lang=%s; query=%s", query_lang, query)

    # Translate to English for parsing, if needed
    if query_lang == "en":
        parse_input = query
        logger.info("[selection-helper] parsing in English (no translation needed)")
    else:
        logger.info("[selection-helper] translating query to English for parsing")
        parse_input = translate_text(query, "en")

    # Build selection prompt with canonical books
    books = ", ".join(books_map.keys())
    system_prompt = selection_prompt_template.format(books=books)
    selection_messages: list[EasyInputMessageParam] = cast(List[EasyInputMessageParam], [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": parse_input},
    ])
    logger.info("[selection-helper] extracting passage selection via LLM")
    selection_resp = open_ai_client.responses.parse(
        model="gpt-4o",
        input=cast(Any, selection_messages),
        text_format=PassageSelection,
        store=False,
    )
    usage = getattr(selection_resp, "usage", None)
    if usage is not None:
        it = getattr(usage, "input_tokens", None)
        ot = getattr(usage, "output_tokens", None)
        tt = getattr(usage, "total_tokens", None)
        if tt is None and (it is not None or ot is not None):
            tt = (it or 0) + (ot or 0)
        cit = extract_cached_input_tokens(usage)
        add_tokens(it, ot, tt, model="gpt-4o", cached_input_tokens=cit)
    selection = cast(Any, selection_resp.output_parsed)
    logger.info("[selection-helper] extracted %d selection(s)", len(selection.selections))

    # Detect books explicitly mentioned in the user input for cross-book guardrails
    mentioned = _detect_mentioned_books(parse_input, books_map)

    # Heuristic correction for explicit "chapters X–Y"
    lower_in = parse_input.lower()
    chap_match = re.search(r"\bchapters?\s+(\d+)\s*[-–]\s*(\d+)\b", lower_in)
    if chap_match and selection.selections:
        a, b = int(chap_match.group(1)), int(chap_match.group(2))
        logger.info("[selection-helper] correcting to multi-chapter range: %d-%d due to 'chapters' phrasing", a, b)
        first = selection.selections[0]
        selection.selections[0] = PassageRef(
            book=first.book,
            start_chapter=a,
            start_verse=None,
            end_chapter=b,
            end_verse=None,
        )

    if not selection.selections:
        if len(mentioned) >= 2:
            msg = (
                "Please request a selection for one book at a time. "
                "If you need multiple books, send a separate message for each."
            )
            logger.info("[selection-helper] empty parse; multiple books detected -> cross-book message")
            return None, None, msg
        if len(mentioned) == 1:
            primary = mentioned[0]
            logger.info("[selection-helper] empty parse; falling back to single detected book: %s", primary)
            return primary, [(1, None, 10_000, None)], None
        msg = (
            "I couldn't identify a clear Bible passage in your request. Supported selection types include: "
            "single verse (e.g., John 3:16); verse range within a chapter (John 3:16-18); cross-chapter within a "
            "single book (John 3:16–4:2); whole chapter (John 3); multi-chapter span with no verses (John 1–4); "
            "or the whole book (John). Multiple books in one request are not supported — please choose one book."
        )
        logger.info("[selection-helper] no passage detected; returning guidance message")
        return None, None, msg

    # Normalize and enforce single-book constraint
    canonical_books: list[str] = []
    normalized_selections: list[Any] = []
    for sel in selection.selections:
        canonical = normalize_book_name(sel.book) or sel.book
        if canonical not in books_map:
            supported = ", ".join(books_map.keys())
            msg = (
                f"The book '{sel.book}' is not recognized. Please use a supported canonical book name. "
                f"Supported books include: {supported}."
            )
            logger.info("[selection-helper] unsupported book requested: %s", sel.book)
            return None, None, msg
        canonical_books.append(canonical)
        normalized_selections.append(PassageRef(
            book=canonical,
            start_chapter=sel.start_chapter,
            start_verse=sel.start_verse,
            end_chapter=sel.end_chapter,
            end_verse=sel.end_verse,
        ))

    if len(set(canonical_books)) != 1:
        msg = (
            "Please request a selection for one book at a time. "
            "If you need multiple books, send a separate message for each."
        )
        logger.info("[selection-helper] cross-book selection detected")
        return None, None, msg

    canonical_book = canonical_books[0]
    logger.info("[selection-helper] canonical_book=%s", canonical_book)

    ranges: list[tuple[int, int | None, int | None, int | None]] = []
    for sel in normalized_selections:
        if sel.start_chapter is None:
            ranges.append((1, None, 10_000, None))
        else:
            ranges.append((sel.start_chapter, sel.start_verse, sel.end_chapter, sel.end_verse))

    logger.info("[selection-helper] ranges=%s", ranges)
    return canonical_book, ranges, None
