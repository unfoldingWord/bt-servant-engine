"""Passage summary intent handler."""
# pylint: disable=line-too-long,too-many-locals,duplicate-code
from __future__ import annotations

from typing import Any, Callable, cast

from openai import OpenAI
from openai.types.responses.easy_input_message_param import (
    EasyInputMessageParam,
)

from logger import get_logger
from servant_brain.classifier import IntentType
from servant_brain.prompts import PASSAGE_SUMMARY_AGENT_SYSTEM_PROMPT
from utils.bible_data import resolve_bible_data_root, load_book_titles
from utils.bible_locale import get_book_name
from utils.bsb import label_ranges, select_verses


logger = get_logger(__name__)


def get_passage_summary(  # noqa: D401 - docstring kept concise here
    state: Any,
    *,
    resolve_selection_for_single_book: Callable[[str, str], tuple[str | None, list[tuple[int, int | None, int | None, int | None]] | None, str | None]],
    open_ai_client: OpenAI,
    add_tokens: Callable[..., None],
    extract_cached_input_tokens: Callable[[Any], int | None],
) -> dict:
    """Handle get-passage-summary: extract refs, retrieve verses, summarize."""
    s = cast(dict[str, Any], state)
    query = cast(str, s["transformed_query"])
    query_lang = cast(str, s["query_language"])
    logger.info("[passage-summary] start; query_lang=%s; query=%s", query_lang, query)

    canonical_book, ranges, err = resolve_selection_for_single_book(query, query_lang)
    if err:
        return {"responses": [{"intent": IntentType.GET_PASSAGE_SUMMARY, "response": err}]}
    assert canonical_book is not None and ranges is not None

    # Resolve source scripture location with fallbacks
    try:
        data_root, resolved_lang, resolved_version = resolve_bible_data_root(
            response_language=s.get("user_response_language"),
            query_language=s.get("query_language"),
            requested_lang=None,
            requested_version=None,
        )
        logger.info(
            "[passage-summary] retrieving verses from %s (lang=%s, version=%s)",
            data_root,
            resolved_lang,
            resolved_version,
        )
    except FileNotFoundError:
        msg = (
            "Scripture data is not available on this server. Please contact the administrator."
        )
        return {"responses": [{"intent": IntentType.GET_PASSAGE_SUMMARY, "response": msg}]}

    verses = select_verses(data_root, canonical_book, ranges)
    logger.info("[passage-summary] retrieved %d verse(s)", len(verses))
    if not verses:
        msg = "I couldn't locate those verses in the Bible data. Please check the reference and try again."
        logger.info("[passage-summary] no verses found for selection; prompting user")
        return {"responses": [{"intent": IntentType.GET_PASSAGE_SUMMARY, "response": msg}]}

    # Build localized reference label
    titles_map = load_book_titles(data_root)
    localized_book = titles_map.get(canonical_book) or get_book_name(str(resolved_lang), canonical_book)
    ref_label_en = label_ranges(canonical_book, ranges)
    if ref_label_en == canonical_book:
        ref_label = localized_book
    elif ref_label_en.startswith(f"{canonical_book} "):
        ref_label = f"{localized_book} {ref_label_en[len(canonical_book) + 1:]}"
    else:
        ref_label = ref_label_en
    logger.info("[passage-summary] label=%s", ref_label)
    joined = "\n".join(f"{ref}: {txt}" for ref, txt in verses)

    # Summarize using LLM with strict system prompt
    sum_messages: list[EasyInputMessageParam] = [
        {"role": "developer", "content": f"Passage reference: {ref_label}"},
        {"role": "developer", "content": f"Passage verses (use only this content):\n{joined}"},
        {"role": "user", "content": "Provide a concise, faithful summary of the passage above."},
    ]
    logger.info("[passage-summary] summarizing %d verses", len(verses))
    summary_resp = open_ai_client.responses.create(
        model="gpt-4o",
        instructions=PASSAGE_SUMMARY_AGENT_SYSTEM_PROMPT,
        input=cast(Any, sum_messages),
        store=False,
    )
    usage = getattr(summary_resp, "usage", None)
    if usage is not None:
        it = getattr(usage, "input_tokens", None)
        ot = getattr(usage, "output_tokens", None)
        tt = getattr(usage, "total_tokens", None)
        if tt is None and (it is not None or ot is not None):
            tt = (it or 0) + (ot or 0)
        cit = extract_cached_input_tokens(usage)
        add_tokens(it, ot, tt, model="gpt-4o", cached_input_tokens=cit)
    summary_text = summary_resp.output_text
    logger.info("[passage-summary] summary generated (len=%d)", len(summary_text) if summary_text else 0)

    response_text = f"Summary of {ref_label}:\n\n{summary_text}"
    logger.info("[passage-summary] done")
    return {"responses": [{"intent": IntentType.GET_PASSAGE_SUMMARY, "response": response_text}]}
