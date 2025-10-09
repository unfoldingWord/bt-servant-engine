"""Intent handlers for passage-based operations (summary, keywords, retrieval, audio)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional, cast

from openai import OpenAI, OpenAIError
from openai.types.responses.easy_input_message_param import EasyInputMessageParam

from bt_servant_engine.core.config import config
from bt_servant_engine.core.intents import IntentType
from bt_servant_engine.core.language import Language, ResponseLanguage
from bt_servant_engine.core.language import SUPPORTED_LANGUAGE_MAP as supported_language_map
from bt_servant_engine.core.logging import get_logger
from bt_servant_engine.services.openai_utils import track_openai_usage
from bt_servant_engine.services.passage_selection import resolve_selection_for_single_book
from utils.bible_data import list_available_sources, load_book_titles, resolve_bible_data_root
from utils.bible_locale import get_book_name
from utils.bsb import label_ranges, select_verses
from utils.keywords import select_keywords
from utils.perf import add_tokens

logger = get_logger(__name__)

PASSAGE_SUMMARY_AGENT_SYSTEM_PROMPT = """
You summarize Bible passage content faithfully using only the verses provided.

- Stay strictly within the supplied passage text; avoid speculation or doctrinal claims not present in the text.
- Highlight the main flow, key ideas, and important movements or contrasts across the entire selection.
- Provide a thorough, readable summary (not terse). Aim for roughly 8–15 sentences, but expand if the selection is large.
- Write in continuous prose only: do NOT use bullets, numbered lists, section headers, or list-like formatting. Compose normal paragraph(s) with sentences flowing naturally.
- Mix verse references inline within the prose wherever helpful (e.g., "1:1–3", "3:16", "2:4–6") to anchor key points rather than isolating them as list items.
- If the selection contains only a single verse, inline verse references are not necessary.
"""

TARGET_TRANSLATION_LANGUAGE_AGENT_SYSTEM_PROMPT = """
Task: Determine the target language the user is asking the system to translate scripture into, based solely on the
user's latest message. Return an ISO 639-1 code from the allowed set.

Allowed outputs: en, ar, fr, es, hi, ru, id, sw, pt, zh, nl, Other

Rules:
- Identify explicit target-language mentions (language names, codes, or phrases like "into Russian", "to es",
  "in French").
- If no target language is explicitly specified, return Other. DO NOT infer a target from the message's language.
- Output must match the provided schema exactly with no extra prose.

Examples:
- message: "translate John 3:16 into Russian" -> { "language": "ru" }
- message: "please translate Mark 1 in Spanish" -> { "language": "es" }
- message: "translate Matthew 2" -> { "language": "Other" }
"""


def get_passage_summary(
    client: OpenAI,
    query: str,
    query_lang: str,
    book_map: dict[str, Any],
    detect_mentioned_books_fn: callable,
    translate_text_fn: callable,
    model_for_agentic_strength_fn: callable,
    extract_cached_input_tokens_fn: callable,
    user_response_language: Optional[str],
    agentic_strength: str,
) -> dict[str, Any]:
    """Handle get-passage-summary: extract refs, retrieve verses, summarize.

    - If user query language is not English, translate the transformed query to English
      for extraction only.
    - Extract passage selection via structured LLM parse with a strict prompt and
      canonical book list.
    - Validate constraints (single book, up to whole book; no cross-book).
    - Load verses from sources/bible_data/en efficiently and summarize.
    - Return a single combined summary prefixed with a canonical reference echo.
    """
    logger.info("[passage-summary] start; query_lang=%s; query=%s", query_lang, query)

    canonical_book, ranges, err = resolve_selection_for_single_book(
        client,
        query,
        query_lang,
        book_map,
        detect_mentioned_books_fn,
        translate_text_fn,
        focus_hint="Focus only on the portion of the user's message that asked for a passage summary. Ignore any other requests or book references in the message.",
    )
    if err:
        return {"responses": [{"intent": IntentType.GET_PASSAGE_SUMMARY, "response": err}]}
    assert canonical_book is not None and ranges is not None

    # Retrieve verses from installed sources (response_language → query_language → en)
    try:
        data_root, resolved_lang, resolved_version = resolve_bible_data_root(
            response_language=user_response_language,
            query_language=query_lang,
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
        msg = "Scripture data is not available on this server. Please contact the administrator."
        return {"responses": [{"intent": IntentType.GET_PASSAGE_SUMMARY, "response": msg}]}

    verses = select_verses(data_root, canonical_book, ranges)
    logger.info("[passage-summary] retrieved %d verse(s)", len(verses))
    if not verses:
        msg = "I couldn't locate those verses in the Bible data. Please check the reference and try again."
        logger.info("[passage-summary] no verses found for selection; prompting user")
        return {"responses": [{"intent": IntentType.GET_PASSAGE_SUMMARY, "response": msg}]}

    # Prepare text for summarization
    # Localize the book name in the header using titles from the resolved source when available
    titles_map = load_book_titles(data_root)
    localized_book = titles_map.get(canonical_book) or get_book_name(
        str(resolved_lang), canonical_book
    )
    ref_label_en = label_ranges(canonical_book, ranges)
    if ref_label_en == canonical_book:
        ref_label = localized_book
    elif ref_label_en.startswith(f"{canonical_book} "):
        ref_label = f"{localized_book} {ref_label_en[len(canonical_book) + 1 :]}"
    else:
        ref_label = ref_label_en
    logger.info("[passage-summary] label=%s", ref_label)
    joined = "\n".join(f"{ref}: {txt}" for ref, txt in verses)

    # Summarize using LLM with strict system prompt
    sum_messages: list[EasyInputMessageParam] = [
        {
            "role": "developer",
            "content": "Focus only on summarizing the portion of the user's message that asked for a passage summary. Ignore any other requests or book references in the message.",
        },
        {"role": "developer", "content": f"Passage reference: {ref_label}"},
        {"role": "developer", "content": f"Passage verses (use only this content):\n{joined}"},
        {"role": "user", "content": "Provide a concise, faithful summary of the passage above."},
    ]
    model_name = model_for_agentic_strength_fn(
        agentic_strength, allow_low=True, allow_very_low=True
    )
    logger.info("[passage-summary] summarizing %d verses", len(verses))
    summary_resp = client.responses.create(
        model=model_name,
        instructions=PASSAGE_SUMMARY_AGENT_SYSTEM_PROMPT,
        input=cast(Any, sum_messages),
        store=False,
    )
    usage = getattr(summary_resp, "usage", None)
    track_openai_usage(usage, model_name, extract_cached_input_tokens_fn, add_tokens)
    summary_text = summary_resp.output_text
    logger.info(
        "[passage-summary] summary generated (len=%d)", len(summary_text) if summary_text else 0
    )

    response_text = f"Summary of {ref_label}:\n\n{summary_text}"
    logger.info("[passage-summary] done")
    return {
        "responses": [
            {
                "intent": IntentType.GET_PASSAGE_SUMMARY,
                "response": response_text,
                "suppress_combining": True,
            }
        ]
    }


def get_passage_keywords(
    client: OpenAI,
    query: str,
    query_lang: str,
    book_map: dict[str, Any],
    detect_mentioned_books_fn: callable,
    translate_text_fn: callable,
) -> dict[str, Any]:
    """Handle get-passage-keywords: extract refs, retrieve keywords, and list them.

    Mirrors the summary flow for selection parsing and validation, but instead of
    summarizing verses, loads per-verse keyword data from sources/keyword_data and
    returns a comma-separated list of distinct tw_match values present in the
    selection. The response is prefixed with "Keywords in <range>\n\n".
    """
    logger.info("[passage-keywords] start; query_lang=%s; query=%s", query_lang, query)

    canonical_book, ranges, err = resolve_selection_for_single_book(
        client,
        query,
        query_lang,
        book_map,
        detect_mentioned_books_fn,
        translate_text_fn,
        focus_hint="Focus only on the portion of the user's message that asked for passage keywords. Ignore any other requests or book references in the message.",
    )
    if err:
        return {"responses": [{"intent": IntentType.GET_PASSAGE_KEYWORDS, "response": err}]}
    assert canonical_book is not None and ranges is not None

    # Retrieve keywords from keyword dataset
    data_root = Path("sources") / "keyword_data"
    logger.info("[passage-keywords] retrieving keywords from %s", data_root)
    keywords = select_keywords(data_root, canonical_book, ranges)
    logger.info("[passage-keywords] retrieved %d keyword(s)", len(keywords))

    if not keywords:
        msg = "I couldn't locate keywords for that selection. Please check the reference and try again."
        logger.info("[passage-keywords] no keywords found; prompting user")
        return {"responses": [{"intent": IntentType.GET_PASSAGE_KEYWORDS, "response": msg}]}

    ref_label = label_ranges(canonical_book, ranges)
    header = f"Keywords in {ref_label}\n\n"
    body = ", ".join(keywords)
    response_text = header + body
    logger.info("[passage-keywords] done")
    return {
        "responses": [
            {
                "intent": IntentType.GET_PASSAGE_KEYWORDS,
                "response": response_text,
                "suppress_combining": True,
            }
        ]
    }


def retrieve_scripture(  # pylint: disable=too-many-arguments,too-many-locals,too-many-branches,too-many-return-statements
    client: OpenAI,
    query: str,
    query_lang: str,
    book_map: dict[str, Any],
    detect_mentioned_books_fn: callable,
    translate_text_fn: callable,
    model_for_agentic_strength_fn: callable,
    extract_cached_input_tokens_fn: callable,
    user_response_language: Optional[str],
    agentic_strength: str,
) -> dict[str, Any]:
    """Handle retrieve-scripture with optional auto-translation.

    Behavior:
    - If the user explicitly requests a particular Bible language (e.g., "in Indonesian"),
      attempt to serve that language verbatim if available. If not available, fall back
      to auto-translating the retrieved scripture into that requested language.
    - Otherwise (no explicit language), retrieve from installed sources using
      response_language → query_language → en for the source. If the chosen source
      language differs from the user's desired response language (or query language
      when response language is unset), auto-translate the scripture into the desired
      target language.
    - Scripture is returned as a structured response with segments. When auto-translating,
      the header is pre-localized and the scripture body is translated while preserving
      line boundaries and verse labels.
    """
    logger.info("[retrieve-scripture] start; query_lang=%s; query=%s", query_lang, query)

    # 1) Parse passage selection
    canonical_book, ranges, err = resolve_selection_for_single_book(
        client,
        query,
        query_lang,
        book_map,
        detect_mentioned_books_fn,
        translate_text_fn,
        focus_hint="Focus only on the portion of the user's message that asked to retrieve or listen to scripture. Ignore any other requests or book references in the message.",
    )
    if err:
        return {"responses": [{"intent": IntentType.RETRIEVE_SCRIPTURE, "response": err}]}
    assert canonical_book is not None and ranges is not None

    # 2) Detect explicit requested source language (e.g., "in Indonesian").
    #    Use the same structured parser used for translate-scripture to keep
    #    language extraction robust, then fall back to a minimal regex.
    requested_lang: Optional[str] = None
    try:
        tl_resp = client.responses.parse(
            model="gpt-4o",
            instructions=TARGET_TRANSLATION_LANGUAGE_AGENT_SYSTEM_PROMPT,
            input=cast(Any, [{"role": "user", "content": f"message: {query}"}]),
            text_format=ResponseLanguage,
            temperature=0,
            store=False,
        )
        tl_usage = getattr(tl_resp, "usage", None)
        track_openai_usage(tl_usage, "gpt-4o", extract_cached_input_tokens_fn, add_tokens)
        tl_parsed = cast(ResponseLanguage | None, tl_resp.output_parsed)
        if tl_parsed and tl_parsed.language != Language.OTHER:
            requested_lang = str(tl_parsed.language.value)
    except OpenAIError:
        logger.info(
            "[retrieve-scripture] requested-language parse failed; will fallback", exc_info=True
        )
    except Exception:  # pylint: disable=broad-except
        logger.info(
            "[retrieve-scripture] requested-language parse failed (generic); will fallback",
            exc_info=True,
        )
    if not requested_lang:
        m = re.search(
            r"\b(?:in|from the|from)\s+([A-Za-z][A-Za-z\- ]{1,30})\b", query, flags=re.IGNORECASE
        )
        if m:
            # Map name → ISO code when possible
            name = m.group(1).strip().title()
            for code, friendly in supported_language_map.items():
                if friendly.lower() == name.lower():
                    requested_lang = code
                    break

    # 3) Resolve bible data root path with fallbacks
    try:
        data_root, resolved_lang, resolved_version = resolve_bible_data_root(
            response_language=user_response_language,
            query_language=query_lang,
            requested_lang=requested_lang,
            requested_version=None,
        )
        logger.info(
            "[retrieve-scripture] data_root=%s lang=%s version=%s",
            data_root,
            resolved_lang,
            resolved_version,
        )
    except FileNotFoundError:
        # Catalog available options for a friendly message
        avail = list_available_sources()
        if not avail:
            msg = (
                "Scripture data is not available on this server. Please contact the administrator."
            )
            return {"responses": [{"intent": IntentType.RETRIEVE_SCRIPTURE, "response": msg}]}
        options = ", ".join(f"{lang}/{ver}" for lang, ver in avail)
        msg = (
            f"I couldn't find a Bible source matching your request. Available sources: {options}. "
            f"Would you like me to use one of these?"
        )
        return {"responses": [{"intent": IntentType.RETRIEVE_SCRIPTURE, "response": msg}]}

    # 4) Enforce verse-count limit before retrieval to avoid oversized selections
    total_verses = len(select_verses(data_root, canonical_book, ranges))
    if total_verses > config.RETRIEVE_SCRIPTURE_VERSE_LIMIT:
        ref_label_over = label_ranges(canonical_book, ranges)
        msg = (
            f"I can only retrieve up to {config.RETRIEVE_SCRIPTURE_VERSE_LIMIT} verses at a time. "
            f"Your selection {ref_label_over} includes {total_verses} verses. "
            "Please narrow the range (e.g., a chapter or a shorter span)."
        )
        return {"responses": [{"intent": IntentType.RETRIEVE_SCRIPTURE, "response": msg}]}

    # 5) Retrieve exact verses (now known to be within limit)
    verses = select_verses(data_root, canonical_book, ranges)
    if not verses:
        msg = "I couldn't locate those verses in the Bible data. Please check the reference and try again."
        return {"responses": [{"intent": IntentType.RETRIEVE_SCRIPTURE, "response": msg}]}

    # 6) Build header segments (book + suffix) and scripture segment
    ref_label = label_ranges(canonical_book, ranges)
    # Derive suffix by removing leading book name, when present
    suffix = ""
    if ref_label == canonical_book:
        suffix = ""
    elif ref_label.startswith(f"{canonical_book} "):
        suffix = ref_label[len(canonical_book) + 1 :]
    else:
        suffix = ref_label
    # Build a flowing paragraph of verse text without chapter:verse labels

    def _norm_ws(s: str) -> str:
        return re.sub(r"\s+", " ", s).strip()

    scripture_lines: list[str] = []
    for _ref, txt in verses:
        scripture_lines.append(_norm_ws(str(txt)))
    # Join with a single space to create a continuous block
    scripture_text = " ".join(scripture_lines)

    # 7) Decide on auto-translation target
    desired_target: Optional[str] = None
    # If the user explicitly requested a language but the resolved source differs,
    # prefer translating to that requested language.
    if requested_lang and requested_lang != resolved_lang:
        desired_target = requested_lang
    # Otherwise, use response_language (if set) or query_language when they differ
    # from the resolved source language.
    if not desired_target and not requested_lang:
        url = cast(Optional[str], user_response_language)
        ql = cast(Optional[str], query_lang)
        target_pref = url or ql
        if target_pref and target_pref != resolved_lang:
            desired_target = target_pref

    # If we have a target and it's a supported language code, auto-translate body + header
    if desired_target and desired_target in supported_language_map:
        # Localize header book name using target language titles when possible;
        # fall back to a static map; finally, LLM-translate as last resort.
        translated_book = None
        try:
            t_root, _t_lang, _t_ver = resolve_bible_data_root(
                response_language=None,
                query_language=None,
                requested_lang=desired_target,
                requested_version=None,
            )
            t_titles = load_book_titles(t_root)
            translated_book = t_titles.get(canonical_book)
        except FileNotFoundError:
            translated_book = None
        if not translated_book:
            static_name = get_book_name(desired_target, canonical_book)
            if static_name != canonical_book:
                translated_book = static_name
            else:
                # As a last resort, translate the canonical book name with the LLM.
                translated_book = translate_text_fn(
                    response_text=canonical_book,
                    target_language=desired_target,
                    agentic_strength=agentic_strength,
                )
        # Translate each verse text and join into a flowing paragraph
        translated_lines: list[str] = [
            _norm_ws(
                translate_text_fn(
                    response_text=str(txt),
                    target_language=desired_target,
                    agentic_strength=agentic_strength,
                )
            )
            for _ref, txt in verses
        ]
        translated_body = " ".join(translated_lines)
        response_obj = {
            "suppress_translation": True,
            "content_language": desired_target,
            "header_is_translated": True,
            "segments": [
                {"type": "header_book", "text": translated_book},
                {"type": "header_suffix", "text": suffix},
                {"type": "scripture", "text": translated_body},
            ],
        }
        return {"responses": [{"intent": IntentType.RETRIEVE_SCRIPTURE, "response": response_obj}]}

    # No auto-translation required; return verbatim with canonical header (to be localized downstream if desired)
    # Load localized titles for the resolved source language (if present)
    titles_map = load_book_titles(data_root)
    header_book = titles_map.get(canonical_book) or get_book_name(
        str(resolved_lang), canonical_book
    )
    response_obj = {
        "suppress_translation": True,
        "content_language": str(resolved_lang),
        "header_is_translated": False,
        "segments": [
            {"type": "header_book", "text": header_book},
            {"type": "header_suffix", "text": suffix},
            {"type": "scripture", "text": scripture_text},
        ],
    }
    return {"responses": [{"intent": IntentType.RETRIEVE_SCRIPTURE, "response": response_obj}]}


def listen_to_scripture(
    client: OpenAI,
    query: str,
    query_lang: str,
    book_map: dict[str, Any],
    detect_mentioned_books_fn: callable,
    translate_text_fn: callable,
    reconstruct_structured_text_fn: callable,
    model_for_agentic_strength_fn: callable,
    extract_cached_input_tokens_fn: callable,
    user_response_language: Optional[str],
    agentic_strength: str,
) -> dict[str, Any]:
    """Delegate to retrieve-scripture and request voice delivery.

    Reuses retrieve-scripture end-to-end (selection, retrieval, translation, formatting)
    and sets a delivery hint that the API should send a voice message.
    """
    out = retrieve_scripture(
        client,
        query,
        query_lang,
        book_map,
        detect_mentioned_books_fn,
        translate_text_fn,
        model_for_agentic_strength_fn,
        extract_cached_input_tokens_fn,
        user_response_language,
        agentic_strength,
    )
    out["send_voice_message"] = True
    responses = cast(list[dict], out.get("responses", []))
    if responses:
        # Reconstruct scripture text for voice playback using the structured response.
        out["voice_message_text"] = reconstruct_structured_text_fn(
            resp_item=responses[0],
            localize_to=None,
        )
        for resp in responses:
            resp["suppress_text_delivery"] = True
    return out


__all__ = [
    "PASSAGE_SUMMARY_AGENT_SYSTEM_PROMPT",
    "TARGET_TRANSLATION_LANGUAGE_AGENT_SYSTEM_PROMPT",
    "get_passage_summary",
    "get_passage_keywords",
    "retrieve_scripture",
    "listen_to_scripture",
]
