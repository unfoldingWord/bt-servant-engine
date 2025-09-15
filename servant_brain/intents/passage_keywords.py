from __future__ import annotations

from typing import Any, Callable, cast

from logger import get_logger
from servant_brain.classifier import IntentType
from utils.bsb import label_ranges
from utils.keywords import select_keywords
from pathlib import Path


logger = get_logger(__name__)


def get_passage_keywords(
    state: Any,
    *,
    resolve_selection_for_single_book: Callable[[str, str], tuple[str | None, list[tuple[int, int | None, int | None, int | None]] | None, str | None]],
) -> dict:
    """Handle get-passage-keywords: extract refs, retrieve keywords, and list them."""
    s = cast(dict[str, Any], state)
    query = cast(str, s["transformed_query"])
    query_lang = cast(str, s["query_language"])
    logger.info("[passage-keywords] start; query_lang=%s; query=%s", query_lang, query)

    canonical_book, ranges, err = resolve_selection_for_single_book(query, query_lang)
    if err:
        return {"responses": [{"intent": IntentType.GET_PASSAGE_KEYWORDS, "response": err}]}
    assert canonical_book is not None and ranges is not None

    # Retrieve keywords from keyword dataset
    data_root = Path("sources") / "keyword_data"
    logger.info("[passage-keywords] retrieving keywords from %s", data_root)
    keywords = select_keywords(data_root, canonical_book, ranges)
    logger.info("[passage-keywords] retrieved %d keyword(s)", len(keywords))

    if not keywords:
        msg = (
            "I couldn't locate keywords for that selection. Please check the reference and try again."
        )
        logger.info("[passage-keywords] no keywords found; prompting user")
        return {"responses": [{"intent": IntentType.GET_PASSAGE_KEYWORDS, "response": msg}]}

    ref_label = label_ranges(canonical_book, ranges)
    header = f"Keywords in {ref_label}\n\n"
    body = ", ".join(keywords)
    response_text = header + body
    logger.info("[passage-keywords] done")
    return {"responses": [{"intent": IntentType.GET_PASSAGE_KEYWORDS, "response": response_text}]}

