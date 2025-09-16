"""Translate scripture intent handler."""
# pylint: disable=line-too-long,too-many-arguments,too-many-locals,too-many-return-statements,duplicate-code
from __future__ import annotations

import re as _re
from typing import Any, Callable, Optional, cast

from openai import OpenAI, OpenAIError
from openai.types.responses.easy_input_message_param import EasyInputMessageParam

from config import config
from logger import get_logger
from servant_brain.classifier import IntentType
from servant_brain.responses import (
    make_scripture_response as _make_scripture_response,
    make_scripture_response_from_translated as _make_scripture_response_from_translated,
    _norm_ws,
)


logger = get_logger(__name__)


def _compute_header_suffix(label_ranges: Callable[[str, Any], str], canonical_book: str, ranges) -> str:  # type: ignore[no-untyped-def]
    ref_label = label_ranges(canonical_book, ranges)
    if ref_label == canonical_book:
        return ""
    if ref_label.startswith(f"{canonical_book} "):
        return ref_label[len(canonical_book) + 1 :]
    return ref_label


def _attempt_structured_translation(
    canonical_book: str,
    header_suffix: str,
    target_code: str,
    body_src: str,
    *,
    open_ai_client: OpenAI,
    add_tokens: Callable[..., None],
    extract_cached_input_tokens: Callable[[Any], int | None],
    TranslatedPassage: Any,
    TRANSLATE_PASSAGE_AGENT_SYSTEM_PROMPT: str,
) -> Any | None:
    try:
        messages: list[EasyInputMessageParam] = [
            {"role": "developer", "content": f"canonical_book: {canonical_book}"},
            {"role": "developer", "content": f"header_suffix (do not translate): {header_suffix}"},
            {"role": "developer", "content": f"target_language: {target_code}"},
            {"role": "developer", "content": "passage body (translate; preserve newlines):"},
            {"role": "developer", "content": body_src},
        ]
        resp = open_ai_client.responses.parse(
            model="gpt-4o",
            instructions=TRANSLATE_PASSAGE_AGENT_SYSTEM_PROMPT,
            input=cast(Any, messages),
            text_format=TranslatedPassage,
            temperature=0,
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
        return cast(Any | None, resp.output_parsed)
    except OpenAIError:
        logger.warning("[translate-scripture] structured parse failed due to OpenAI error; falling back.", exc_info=True)
    except Exception:  # pylint: disable=broad-except
        logger.warning("[translate-scripture] structured parse failed; falling back to simple translation.", exc_info=True)
    return None


def _resolve_target_language(
    query: str,
    *,
    open_ai_client: OpenAI,
    add_tokens: Callable[..., None],
    extract_cached_input_tokens: Callable[[Any], int | None],
    ResponseLanguage: Any,
    TARGET_TRANSLATION_LANGUAGE_AGENT_SYSTEM_PROMPT: str,
) -> tuple[Optional[str], Optional[str]]:
    target_code: Optional[str] = None
    explicit_mention_name: Optional[str] = None
    try:
        tl_resp = open_ai_client.responses.parse(
            model="gpt-4o",
            instructions=TARGET_TRANSLATION_LANGUAGE_AGENT_SYSTEM_PROMPT,
            input=cast(Any, [{"role": "user", "content": f"message: {query}"}]),
            text_format=ResponseLanguage,
            temperature=0,
            store=False,
        )
        tl_usage = getattr(tl_resp, "usage", None)
        if tl_usage is not None:
            it = getattr(tl_usage, "input_tokens", None)
            ot = getattr(tl_usage, "output_tokens", None)
            tt = getattr(tl_usage, "total_tokens", None)
            if tt is None and (it is not None or ot is not None):
                tt = (it or 0) + (ot or 0)
            cit = extract_cached_input_tokens(tl_usage)
            add_tokens(it, ot, tt, model="gpt-4o", cached_input_tokens=cit)
        tl_parsed = cast(Any | None, tl_resp.output_parsed)
        if tl_parsed and getattr(tl_parsed, "language", None) not in (None, "Other"):
            target_code = str(tl_parsed.language.value)
    except OpenAIError:
        logger.info("[translate-scripture] target-language parse failed; will fallback", exc_info=True)

    if not target_code:
        m = _re.search(r"\b(?:into|to|in)\s+([A-Za-z][A-Za-z\- ]{1,30})\b", query, flags=_re.IGNORECASE)
        if m:
            explicit_mention_name = m.group(1).strip().title()

    return target_code, explicit_mention_name


def _fallback_target_language(
    state: dict[str, Any],
    *,
    supported_language_map: dict[str, str],
) -> Optional[str]:
    url = cast(Optional[str], state.get("user_response_language"))
    if url and url in supported_language_map:
        return url
    ql = cast(Optional[str], state.get("query_language"))
    if ql and ql in supported_language_map:
        return ql
    return None


def _supported_language_guidance(requested_name: Optional[str], *, supported_language_map: dict[str, str]) -> str:
    name = requested_name or "an unsupported language"
    supported_names = [
        supported_language_map[code]
        for code in ["en", "ar", "fr", "es", "hi", "ru", "id", "sw", "pt", "zh", "nl"]
    ]
    supported_lines = "\n".join(f"- {n}" for n in supported_names)
    return (
        f"Translating into {name} is currently not supported.\n\n"
        "BT Servant can set your response language to any of:\n\n"
        f"{supported_lines}\n\n"
        "Would you like me to set a specific language for your responses?"
    )


def translate_scripture(  # noqa: C901
    state: Any,
    *,
    resolve_selection_for_single_book: Callable[[str, str], tuple[str | None, list[tuple[int, int | None, int | None, int | None]] | None, str | None]],
    resolve_bible_data_root: Callable[..., Any],
    list_available_sources: Callable[[], list[tuple[str, str]]],
    select_verses: Callable[..., list[tuple[str, str]]],
    label_ranges: Callable[[str, Any], str],
    translate_text: Callable[[str, str], str],
    open_ai_client: OpenAI,
    add_tokens: Callable[..., None],
    extract_cached_input_tokens: Callable[[Any], int | None],
    TranslatedPassage: Any,
    ResponseLanguage: Any,
    TARGET_TRANSLATION_LANGUAGE_AGENT_SYSTEM_PROMPT: str,
    TRANSLATE_PASSAGE_AGENT_SYSTEM_PROMPT: str,
    supported_language_map: dict[str, str],
) -> dict:
    """Handle translate-scripture: return verses translated into a target language."""
    s = cast(dict[str, Any], state)
    query = cast(str, s["transformed_query"])
    query_lang = cast(str, s["query_language"])
    logger.info("[translate-scripture] start; query_lang=%s; query=%s", query_lang, query)

    # Validate passage selection first
    canonical_book, ranges, err = resolve_selection_for_single_book(query, query_lang)
    if err:
        return {"responses": [{"intent": IntentType.TRANSLATE_SCRIPTURE, "response": err}]}
    assert canonical_book is not None and ranges is not None

    # Resolve target language and handle unsupported explicit requests
    target_code, explicit_name = _resolve_target_language(
        query,
        open_ai_client=open_ai_client,
        add_tokens=add_tokens,
        extract_cached_input_tokens=extract_cached_input_tokens,
        ResponseLanguage=ResponseLanguage,
        TARGET_TRANSLATION_LANGUAGE_AGENT_SYSTEM_PROMPT=TARGET_TRANSLATION_LANGUAGE_AGENT_SYSTEM_PROMPT,
    )
    if not target_code and explicit_name:
        guidance = _supported_language_guidance(explicit_name, supported_language_map=supported_language_map)
        return {"responses": [{"intent": IntentType.TRANSLATE_SCRIPTURE, "response": guidance}]}
    if not target_code:
        target_code = _fallback_target_language(s, supported_language_map=supported_language_map)
    if not target_code or target_code not in supported_language_map:
        guidance = _supported_language_guidance(explicit_name, supported_language_map=supported_language_map)
        return {"responses": [{"intent": IntentType.TRANSLATE_SCRIPTURE, "response": guidance}]}

    # Resolve source Bible data
    try:
        src = resolve_bible_data_root(
            response_language=s.get("user_response_language"),
            query_language=s.get("query_language"),
            requested_lang=None,
            requested_version=None,
        )
        data_root, resolved_lang, _resolved_version = src
        logger.info(
            "[translate-scripture] source data_root=%s lang=%s version=%s",
            data_root,
            resolved_lang,
            _resolved_version,
        )
    except FileNotFoundError:
        avail = list_available_sources()
        if not avail:
            return {"responses": [{"intent": IntentType.TRANSLATE_SCRIPTURE, "response": "Scripture data is not available on this server. Please contact the administrator."}]}
        options = ", ".join(f"{lang}/{ver}" for lang, ver in avail)
        return {"responses": [{"intent": IntentType.TRANSLATE_SCRIPTURE, "response": f"I couldn't find a Bible source to translate from. Available sources: {options}. Would you like me to use one of these?"}]}

    # Enforce verse-count limit and retrieve verses
    total_verses = len(select_verses(data_root, canonical_book, ranges))
    if total_verses > config.TRANSLATE_SCRIPTURE_VERSE_LIMIT:
        ref_label_over = label_ranges(canonical_book, ranges)
        msg = (
            f"I can only translate up to {config.TRANSLATE_SCRIPTURE_VERSE_LIMIT} verses at a time. "
            f"Your selection {ref_label_over} includes {total_verses} verses. "
            "Please narrow the range (e.g., a chapter or a shorter span)."
        )
        return {"responses": [{"intent": IntentType.TRANSLATE_SCRIPTURE, "response": msg}]}

    verses = select_verses(data_root, canonical_book, ranges)
    if not verses:
        msg = "I couldn't locate those verses in the Bible data. Please check the reference and try again."
        return {"responses": [{"intent": IntentType.TRANSLATE_SCRIPTURE, "response": msg}]}

    header_suffix = _compute_header_suffix(label_ranges, canonical_book, ranges)
    body_src = " ".join(_norm_ws(txt) for _, txt in verses)

    # If target is same as source, return verbatim
    if target_code == resolved_lang:
        response_obj = _make_scripture_response(
            content_language=str(resolved_lang),
            header_book=canonical_book,
            header_suffix=header_suffix,
            scripture_text=body_src,
            header_is_translated=False,
        )
        return {"responses": [{"intent": IntentType.TRANSLATE_SCRIPTURE, "response": response_obj}]}

    # Try structured translation; fall back to simple text translation
    translated = _attempt_structured_translation(
        canonical_book,
        header_suffix,
        cast(str, target_code),
        body_src,
        open_ai_client=open_ai_client,
        add_tokens=add_tokens,
        extract_cached_input_tokens=extract_cached_input_tokens,
        TranslatedPassage=TranslatedPassage,
        TRANSLATE_PASSAGE_AGENT_SYSTEM_PROMPT=TRANSLATE_PASSAGE_AGENT_SYSTEM_PROMPT,
    )
    if translated is None:
        translated_body = _norm_ws(translate_text(body_src, cast(str, target_code)))
        translated_book = translate_text(canonical_book, cast(str, target_code))
        response_obj = _make_scripture_response(
            content_language=cast(str, target_code),
            header_book=translated_book,
            header_suffix=header_suffix,
            scripture_text=translated_body,
            header_is_translated=True,
        )
    else:
        response_obj = _make_scripture_response_from_translated(
            canonical_book=canonical_book,
            header_suffix=header_suffix,
            translated=translated,
        )
    return {"responses": [{"intent": IntentType.TRANSLATE_SCRIPTURE, "response": response_obj}]}
