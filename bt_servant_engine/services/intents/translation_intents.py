"""Intent handlers for scripture translation and translation helps."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional, cast

from openai import OpenAI, OpenAIError
from openai.types.responses.easy_input_message_param import EasyInputMessageParam

from bt_servant_engine.core.config import config
from bt_servant_engine.core.intents import IntentType
from bt_servant_engine.core.language import Language, ResponseLanguage, TranslatedPassage
from bt_servant_engine.core.language import SUPPORTED_LANGUAGE_MAP as supported_language_map
from bt_servant_engine.core.logging import get_logger
from bt_servant_engine.services.openai_utils import track_openai_usage
from bt_servant_engine.services.passage_selection import resolve_selection_for_single_book
from utils.bible_data import list_available_sources, resolve_bible_data_root
from utils.bsb import label_ranges, select_verses
from utils.perf import add_tokens

logger = get_logger(__name__)

TRANSLATE_PASSAGE_AGENT_SYSTEM_PROMPT = """
# Task

Translate the provided scripture passage into the specified target language and return a STRICT JSON object
matching the provided schema. Do not include any extra prose, commentary, code fences, or formatting.

# Rules
- header_book: translate ONLY the canonical book name into the target language (e.g., "John" -> "Иоанн").
- header_suffix: DO NOT translate or alter; copy exactly the provided suffix (e.g., "1:1–7").
- body: translate the passage body into the target language; PRESERVE all newline boundaries exactly; do not add
  bullets, numbers, verse labels, or extra headings.
- content_language: the ISO 639-1 code of the target language.

# Output
Return JSON matching the schema with fields: header_book, header_suffix, body, content_language. No extra keys.
"""

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
- Cite verse numbers inline (e.g., "1:1–3", "3:16") where helpful.
- Be faithful and restrained; do not speculate beyond the provided context.
"""


def translate_scripture(  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals,too-many-branches,too-many-return-statements
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
    """Handle translate-scripture: return verses translated into a target language.

    - Extract passage selection via the shared helper.
    - Determine target language from the user's message; require it to be one of supported codes.
    - Optional source language/version parsing (simple language-name heuristic); if absent, use resolver fallbacks.
    - Load source verse texts, translate only the verse text per line, and return a structured, protected response.
    """
    logger.info("[translate-scripture] start; query_lang=%s; query=%s", query_lang, query)

    # First, validate the passage selection so we can surface selection errors
    # (e.g., unsupported book like "Enoch") before language guidance.
    canonical_book, ranges, err = resolve_selection_for_single_book(
        client,
        query,
        query_lang,
        book_map,
        detect_mentioned_books_fn,
        translate_text_fn,
        focus_hint="Focus only on the portion of the user's message that asked to translate scripture. Ignore any other requests or book references in the message.",
    )
    if err:
        return {"responses": [{"intent": IntentType.TRANSLATE_SCRIPTURE, "response": err}]}
    assert canonical_book is not None and ranges is not None

    # Determine target language for translation
    # 1) Try to extract an explicit target from the message via structured parse
    target_code: Optional[str] = None
    explicit_mention_name: Optional[str] = None
    try:
        from bt_servant_engine.services.intents.passage_intents import (
            TARGET_TRANSLATION_LANGUAGE_AGENT_SYSTEM_PROMPT,
        )

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
            target_code = str(tl_parsed.language.value)
    except OpenAIError:
        logger.info(
            "[translate-scripture] target-language parse failed; will fallback", exc_info=True
        )

    # Minimal, tightly scoped heuristic for phrases like "into Italian" (explicit mention)
    if not target_code:
        m = re.search(
            r"\b(?:into|to|in)\s+([A-Za-z][A-Za-z\- ]{1,30})\b", query, flags=re.IGNORECASE
        )
        if m:
            explicit_mention_name = m.group(1).strip().title()

    # 2) Decide path: if explicit mention exists but unsupported -> guidance; else fall back.
    if not target_code and explicit_mention_name:
        # Explicitly requested a language, but it's not in our supported codes.
        requested_name = explicit_mention_name
        supported_names = [
            supported_language_map[c]
            for c in ["en", "ar", "fr", "es", "hi", "ru", "id", "sw", "pt", "zh", "nl"]
        ]
        supported_lines = "\n".join(f"- {name}" for name in supported_names)
        guidance = (
            f"Translating into {requested_name} is currently not supported.\n\n"
            "BT Servant can set your response language to any of:\n\n"
            f"{supported_lines}\n\n"
            "Would you like me to set a specific language for your responses?"
        )
        return {"responses": [{"intent": IntentType.TRANSLATE_SCRIPTURE, "response": guidance}]}

    # 3) Fallbacks: user_response_language, then detected query_language
    if not target_code:
        url = user_response_language
        if url and url in supported_language_map:
            target_code = url
    if not target_code:
        ql = query_lang
        if ql and ql in supported_language_map:
            target_code = ql

    # If we still don't know a supported target, return guidance with supported list
    if not target_code or target_code not in supported_language_map:
        # Prefer explicit language name if present in the message for clarity
        requested_name2: Optional[str] = explicit_mention_name
        if not requested_name2:
            requested_name2 = "an unsupported language"

        supported_names = [
            supported_language_map[code]
            for code in ["en", "ar", "fr", "es", "hi", "ru", "id", "sw", "pt", "zh", "nl"]
        ]
        supported_lines = "\n".join(f"- {name}" for name in supported_names)
        guidance = (
            f"Translating into {requested_name2} is currently not supported.\n\n"
            "BT Servant can set your response language to any of:\n\n"
            f"{supported_lines}\n\n"
            "Would you like me to set a specific language for your responses?"
        )
        return {"responses": [{"intent": IntentType.TRANSLATE_SCRIPTURE, "response": guidance}]}

    # Resolve source Bible data (language/version) for retrieval
    try:
        data_root, resolved_lang, resolved_version = resolve_bible_data_root(
            response_language=user_response_language,
            query_language=query_lang,
            requested_lang=None,
            requested_version=None,
        )
        logger.info(
            "[translate-scripture] source data_root=%s lang=%s version=%s",
            data_root,
            resolved_lang,
            resolved_version,
        )
    except FileNotFoundError:
        avail = list_available_sources()
        if not avail:
            msg = (
                "Scripture data is not available on this server. Please contact the administrator."
            )
            return {"responses": [{"intent": IntentType.TRANSLATE_SCRIPTURE, "response": msg}]}
        options = ", ".join(f"{lang}/{ver}" for lang, ver in avail)
        msg = (
            f"I couldn't find a Bible source to translate from. Available sources: {options}. "
            f"Would you like me to use one of these?"
        )
        return {"responses": [{"intent": IntentType.TRANSLATE_SCRIPTURE, "response": msg}]}

    # Enforce verse-count limit before retrieval/translation to avoid oversized selections
    total_verses = len(select_verses(data_root, canonical_book, ranges))
    if total_verses > config.TRANSLATE_SCRIPTURE_VERSE_LIMIT:
        ref_label_over = label_ranges(canonical_book, ranges)
        msg = (
            f"I can only translate up to {config.TRANSLATE_SCRIPTURE_VERSE_LIMIT} verses at a time. "
            f"Your selection {ref_label_over} includes {total_verses} verses. "
            "Please narrow the range (e.g., a chapter or a shorter span)."
        )
        return {"responses": [{"intent": IntentType.TRANSLATE_SCRIPTURE, "response": msg}]}

    # Retrieve verses
    verses = select_verses(data_root, canonical_book, ranges)
    if not verses:
        msg = "I couldn't locate those verses in the Bible data. Please check the reference and try again."
        return {"responses": [{"intent": IntentType.TRANSLATE_SCRIPTURE, "response": msg}]}

    # Build header suffix once from the canonical label
    ref_label = label_ranges(canonical_book, ranges)
    if ref_label == canonical_book:
        header_suffix = ""
    elif ref_label.startswith(f"{canonical_book} "):
        header_suffix = ref_label[len(canonical_book) + 1 :]
    else:
        header_suffix = ref_label

    # Join body as continuous text (drop verse labels; single paragraph)
    def _norm_ws(s: str) -> str:
        return re.sub(r"\s+", " ", str(s)).strip()

    body_src = " ".join(_norm_ws(txt) for _, txt in verses)

    # Pass-through guard: if source language equals target language, return verbatim
    # scripture without running translation. This avoids unnecessary LLM calls and
    # preserves the original text (e.g., source=fr and target=fr).
    if target_code and target_code == resolved_lang:
        response_obj = {
            "suppress_translation": True,
            "content_language": str(resolved_lang),
            "header_is_translated": False,
            "segments": [
                {"type": "header_book", "text": canonical_book},
                {"type": "header_suffix", "text": header_suffix},
                {"type": "scripture", "text": body_src},
            ],
        }
        return {"responses": [{"intent": IntentType.TRANSLATE_SCRIPTURE, "response": response_obj}]}

    # Attempt structured translation (book header + body) via Responses.parse
    translated: TranslatedPassage | None = None
    try:
        messages: list[EasyInputMessageParam] = [
            {"role": "developer", "content": f"canonical_book: {canonical_book}"},
            {"role": "developer", "content": f"header_suffix (do not translate): {header_suffix}"},
            {"role": "developer", "content": f"target_language: {target_code}"},
            {"role": "developer", "content": "passage body (translate; preserve newlines):"},
            {"role": "developer", "content": body_src},
        ]
        model_name = model_for_agentic_strength_fn(
            agentic_strength, allow_low=False, allow_very_low=True
        )
        resp = client.responses.parse(
            model=model_name,
            instructions=TRANSLATE_PASSAGE_AGENT_SYSTEM_PROMPT,
            input=cast(Any, messages),
            text_format=TranslatedPassage,
            temperature=0,
            store=False,
        )
        usage = getattr(resp, "usage", None)
        track_openai_usage(usage, model_name, extract_cached_input_tokens_fn, add_tokens)
        translated = cast(TranslatedPassage | None, resp.output_parsed)
    except OpenAIError:
        logger.warning(
            "[translate-scripture] structured parse failed due to OpenAI error; falling back.",
            exc_info=True,
        )
        translated = None
    except Exception:  # pylint: disable=broad-except
        logger.warning(
            "[translate-scripture] structured parse failed; falling back to simple translation.",
            exc_info=True,
        )
        translated = None

    if translated is None:
        translated_body = _norm_ws(
            translate_text_fn(
                response_text=body_src,
                target_language=cast(str, target_code),
                agentic_strength=agentic_strength,
            )
        )
        translated_book = translate_text_fn(
            response_text=canonical_book,
            target_language=cast(str, target_code),
            agentic_strength=agentic_strength,
        )
        response_obj = {
            "suppress_translation": True,
            "content_language": cast(str, target_code),
            "header_is_translated": True,
            "segments": [
                {"type": "header_book", "text": translated_book},
                {"type": "header_suffix", "text": header_suffix},
                {"type": "scripture", "text": translated_body},
            ],
        }
    else:
        response_obj = {
            "suppress_translation": True,
            "content_language": str(translated.content_language.value),
            "header_is_translated": True,
            "segments": [
                {"type": "header_book", "text": translated.header_book or canonical_book},
                {"type": "header_suffix", "text": translated.header_suffix or header_suffix},
                {"type": "scripture", "text": _norm_ws(translated.body)},
            ],
        }
    return {"responses": [{"intent": IntentType.TRANSLATE_SCRIPTURE, "response": response_obj}]}


def get_translation_helps(
    client: OpenAI,
    query: str,
    query_lang: str,
    book_map: dict[str, Any],
    detect_mentioned_books_fn: callable,
    translate_text_fn: callable,
    model_for_agentic_strength_fn: callable,
    extract_cached_input_tokens_fn: callable,
    prepare_translation_helps_fn: callable,
    build_translation_helps_context_fn: callable,
    build_translation_helps_messages_fn: callable,
    agentic_strength: str,
) -> dict[str, Any]:
    """Generate focused translation helps guidance for a selected passage."""
    logger.info("[translation-helps] start; query_lang=%s; query=%s", query_lang, query)

    th_root = Path("sources") / "translation_helps"
    bsb_root = Path("sources") / "bible_data" / "en" / "bsb"
    logger.info("[translation-helps] loading helps from %s", th_root)

    canonical_book, ranges, raw_helps, err = prepare_translation_helps_fn(
        client,
        query,
        query_lang,
        th_root,
        bsb_root,
        book_map=book_map,
        detect_mentioned_books_fn=detect_mentioned_books_fn,
        translate_text_fn=translate_text_fn,
        selection_focus_hint="Focus only on the portion of the user's message that asked for translation helps. Ignore any other requests or book references in the message.",
    )
    if err:
        return {"responses": [{"intent": IntentType.GET_TRANSLATION_HELPS, "response": err}]}
    assert canonical_book is not None and ranges is not None and raw_helps is not None

    ref_label, context_obj = build_translation_helps_context_fn(canonical_book, ranges, raw_helps)
    messages = build_translation_helps_messages_fn(ref_label, context_obj)

    logger.info("[translation-helps] invoking LLM with %d helps", len(raw_helps))
    model_name = model_for_agentic_strength_fn(
        agentic_strength, allow_low=True, allow_very_low=True
    )
    resp = client.responses.create(
        model=model_name,
        instructions=TRANSLATION_HELPS_AGENT_SYSTEM_PROMPT,
        input=cast(Any, messages),
        store=False,
    )
    usage = getattr(resp, "usage", None)
    track_openai_usage(usage, model_name, extract_cached_input_tokens_fn, add_tokens)

    header = f"Translation helps for {ref_label}\n\n"
    response_text = header + (resp.output_text or "")
    return {
        "responses": [
            {
                "intent": IntentType.GET_TRANSLATION_HELPS,
                "response": response_text,
                "suppress_combining": True,
            }
        ]
    }


__all__ = [
    "TRANSLATE_PASSAGE_AGENT_SYSTEM_PROMPT",
    "TRANSLATION_HELPS_AGENT_SYSTEM_PROMPT",
    "translate_scripture",
    "get_translation_helps",
]
