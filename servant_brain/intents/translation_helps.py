"""Translation helps intent handler."""
# pylint: disable=line-too-long,too-many-locals,duplicate-code
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, cast

from openai import OpenAI
from openai.types.responses.easy_input_message_param import (
    EasyInputMessageParam,
)

from config import config
from logger import get_logger
from servant_brain.classifier import IntentType
from utils.bsb import (
    BOOK_MAP as BSB_BOOK_MAP,
    label_ranges,
    select_verses,
    clamp_ranges_by_verse_limit,
)
from utils.translation_helps import select_translation_helps, get_missing_th_books


logger = get_logger(__name__)


TRANSLATION_HELPS_AGENT_SYSTEM_PROMPT = """
# Identity

You are a careful assistant helping Bible translators anticipate and address translation issues.

# Instructions

You will receive a structured JSON context containing:
- selection metadata (book and ranges),
- per-verse translation helps (with BSB/ULT verse text and notes).

Use only the provided context to write a coherent, actionable guide for translators. Focus on:
- key translation issues surfaced by the notes,
- clarifications about original-language expressions noted in the helps,
- concrete guidance and options for difficult terms, and
- any cross-references or constraints hinted by support references.

Style:
- Write in clear prose (avoid lists unless the content is inherently a short list).
- Cite verse numbers inline (e.g., “1:1–3”, “3:16”) where helpful.
- Be faithful and restrained; do not speculate beyond the provided context.
"""


def _make_translation_helps_context(
    canonical_book: str,
    limited_ranges,  # type: ignore[no-untyped-def]
    helps: list[dict],
) -> tuple[str, dict]:
    ref_label = label_ranges(canonical_book, limited_ranges)
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
                for (sc, sv, ec, ev) in limited_ranges
            ],
        },
        "translation_helps": helps,
    }
    return ref_label, context_obj


def _run_translation_helps_llm(
    ref_label: str,
    context_obj: dict,
    helps_count: int,
    *,
    open_ai_client: OpenAI,
    add_tokens: Callable[..., None],
    extract_cached_input_tokens: Callable[[Any], int | None],
) -> str:
    messages: list[EasyInputMessageParam] = [
        {"role": "developer", "content": f"Selection: {ref_label}"},
        {"role": "developer", "content": "Use the JSON context below strictly:"},
        {"role": "developer", "content": json.dumps(context_obj, ensure_ascii=False)},
        {"role": "user", "content": "Using the provided context, explain the translation challenges and give actionable guidance for this selection."},
    ]
    logger.info("[translation-helps] invoking LLM with %d helps", helps_count)
    resp = open_ai_client.responses.create(
        model="gpt-4o",
        instructions=TRANSLATION_HELPS_AGENT_SYSTEM_PROMPT,
        input=cast(Any, messages),
        store=False,
    )
    usage = getattr(resp, "usage", None)
    if usage is not None:
        it = getattr(usage, "input_tokens", None)
        ot = getattr(usage, "output_tokens", None)
        tt = getattr(usage, "total_tokens", None)
        if tt is None and (it is not None or ot is not None):
            tt = (it or 0) + (ot or 0)
        cit = extract_cached_input_tokens(usage)
        add_tokens(it, ot, tt, model="gpt-4o", cached_input_tokens=cit)
    text = resp.output_text
    header = f"Translation helps for {ref_label}\n\n"
    return header + (text or "")


def get_translation_helps(
    state: Any,
    *,
    resolve_selection_for_single_book: Callable[[str, str], tuple[str | None, list[tuple[int, int | None, int | None, int | None]] | None, str | None]],
    open_ai_client: OpenAI,
    add_tokens: Callable[..., None],
    extract_cached_input_tokens: Callable[[Any], int | None],
) -> dict:
    """Handle get-translation-helps: extract refs, load helps, and guide."""
    s = cast(dict[str, Any], state)
    query = cast(str, s["transformed_query"])
    query_lang = cast(str, s["query_language"])
    logger.info("[translation-helps] start; query_lang=%s; query=%s", query_lang, query)

    canonical_book, ranges, err = resolve_selection_for_single_book(query, query_lang)
    if err:
        return {"responses": [{"intent": IntentType.GET_TRANSLATION_HELPS, "response": err}]}
    assert canonical_book is not None and ranges is not None

    th_root = Path("sources") / "translation_helps"
    logger.info("[translation-helps] loading helps from %s", th_root)

    # Early exit if the entire book is missing from TH dataset
    missing_books = set(get_missing_th_books(th_root))
    if canonical_book in missing_books:
        abbrs = sorted(BSB_BOOK_MAP[b]["ref_abbr"] for b in missing_books)
        requested_abbr = BSB_BOOK_MAP[canonical_book]["ref_abbr"]
        msg = (
            f"Translation helps for {requested_abbr} are not available yet. "
            f"Currently missing books: {', '.join(abbrs)}. "
            "Would you like translation help for one of the supported books instead?"
        )
        return {"responses": [{"intent": IntentType.GET_TRANSLATION_HELPS, "response": msg}]}

    # Enforce verse limit and clamp selection
    bsb_root = Path("sources") / "bible_data" / "en" / "bsb"
    verse_count = len(select_verses(bsb_root, canonical_book, ranges))
    if verse_count > config.TRANSLATION_HELPS_VERSE_LIMIT:
        ref_label_over = label_ranges(canonical_book, ranges)
        msg = (
            f"I can only provide translate help for {config.TRANSLATION_HELPS_VERSE_LIMIT} verses at a time. "
            f"Your selection {ref_label_over} includes {verse_count} verses. Please narrow the range (e.g., a chapter or a shorter span)."
        )
        return {"responses": [{"intent": IntentType.GET_TRANSLATION_HELPS, "response": msg}]}
    limited_ranges = clamp_ranges_by_verse_limit(
        bsb_root,
        canonical_book,
        ranges,
        max_verses=config.TRANSLATION_HELPS_VERSE_LIMIT,
    )
    if not limited_ranges:
        msg = "I couldn't identify verses for that selection in the BSB index. Please try another reference."
        return {"responses": [{"intent": IntentType.GET_TRANSLATION_HELPS, "response": msg}]}

    helps = select_translation_helps(th_root, canonical_book, limited_ranges)
    logger.info("[translation-helps] selected %d help entries", len(helps))
    if not helps:
        msg = (
            "I couldn't locate translation helps for that selection. Please check the reference and try again."
        )
        return {"responses": [{"intent": IntentType.GET_TRANSLATION_HELPS, "response": msg}]}

    ref_label, context_obj = _make_translation_helps_context(canonical_book, limited_ranges, helps)
    response_text = _run_translation_helps_llm(
        ref_label,
        context_obj,
        len(helps),
        open_ai_client=open_ai_client,
        add_tokens=add_tokens,
        extract_cached_input_tokens=extract_cached_input_tokens,
    )
    return {"responses": [{"intent": IntentType.GET_TRANSLATION_HELPS, "response": response_text}]}
