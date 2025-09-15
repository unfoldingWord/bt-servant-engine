from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable, Optional, cast

from openai import OpenAI, OpenAIError

from config import config
from logger import get_logger
from servant_brain.classifier import IntentType
from servant_brain.responses import (
    make_scripture_response as _make_scripture_response,
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


def _resolve_requested_source_language(
    query: str,
    *,
    open_ai_client: OpenAI,
    add_tokens: Callable[..., None],
    extract_cached_input_tokens: Callable[[Any], int | None],
    ResponseLanguage: Any,
    Language: Any,
    TARGET_TRANSLATION_LANGUAGE_AGENT_SYSTEM_PROMPT: str,
    supported_language_map: dict[str, str],
) -> Optional[str]:
    requested_lang: Optional[str] = None
    try:
        tl_resp = open_ai_client.responses.parse(
            model="gpt-4o",
            instructions=TARGET_TRANSLATION_LANGUAGE_AGENT_SYSTEM_PROMPT,
            input=cast(Any, [{"role": "user", "content": f"message: {query}"}]),
            text_format=ResponseLanguage,
            temperature=0,
            store=False,
        )
        tl_parsed = cast(Optional[Any], tl_resp.output_parsed)
        tl_usage = getattr(tl_resp, "usage", None)
        if tl_usage is not None:
            it = getattr(tl_usage, "input_tokens", None)
            ot = getattr(tl_usage, "output_tokens", None)
            tt = getattr(tl_usage, "total_tokens", None)
            if tt is None and (it is not None or ot is not None):
                tt = (it or 0) + (ot or 0)
            cit = extract_cached_input_tokens(tl_usage)
            add_tokens(it, ot, tt, model="gpt-4o", cached_input_tokens=cit)
        if tl_parsed and tl_parsed.language != Language.OTHER:
            requested_lang = str(tl_parsed.language.value)
    except OpenAIError:
        logger.info("[retrieve-scripture] requested-language parse failed; will fallback", exc_info=True)
    except Exception:  # pylint: disable=broad-except
        logger.info("[retrieve-scripture] requested-language parse failed (generic); will fallback", exc_info=True)
    if not requested_lang:
        m = re.search(r"\b(?:in|from the|from)\s+([A-Za-z][A-Za-z\- ]{1,30})\b", query, flags=re.IGNORECASE)
        if m:
            name = m.group(1).strip().title()
            for code, friendly in supported_language_map.items():
                if friendly.lower() == name.lower():
                    requested_lang = code
                    break
    return requested_lang


def _resolve_source_bible_with_requested(
    *,
    state: dict[str, Any],
    requested_lang: Optional[str],
    resolve_bible_data_root: Callable[..., Any],
    list_available_sources: Callable[[], list[tuple[str, str]]],
    logger_=logger,
):
    try:
        data_root, resolved_lang, resolved_version = resolve_bible_data_root(
            response_language=state.get("user_response_language"),
            query_language=state.get("query_language"),
            requested_lang=requested_lang,
            requested_version=None,
        )
        logger_.info(
            "[retrieve-scripture] data_root=%s lang=%s version=%s",
            data_root,
            resolved_lang,
            resolved_version,
        )
        return data_root, resolved_lang, resolved_version
    except FileNotFoundError:
        avail = list_available_sources()
        if not avail:
            return "Scripture data is not available on this server. Please contact the administrator."
        options = ", ".join(f"{lang}/{ver}" for lang, ver in avail)
        return (
            f"I couldn't find a Bible source matching your request. Available sources: {options}. "
            f"Would you like me to use one of these?"
        )


def _enforce_retrieve_limit_and_select(
    *,
    data_root: Path,
    canonical_book: str,
    ranges,
    select_verses: Callable[..., list[tuple[str, str]]],
    label_ranges: Callable[[str, Any], str],
) -> list[tuple[str, str]] | str:
    total_verses = len(select_verses(data_root, canonical_book, ranges))
    if total_verses > config.RETRIEVE_SCRIPTURE_VERSE_LIMIT:
        ref_label_over = label_ranges(canonical_book, ranges)
        return (
            f"I can only retrieve up to {config.RETRIEVE_SCRIPTURE_VERSE_LIMIT} verses at a time. "
            f"Your selection {ref_label_over} includes {total_verses} verses. "
            "Please narrow the range (e.g., a chapter or a shorter span)."
        )
    verses = select_verses(data_root, canonical_book, ranges)
    if not verses:
        return "I couldn't locate those verses in the Bible data. Please check the reference and try again."
    return verses


def _decide_retrieve_target(
    *,
    state: dict[str, Any],
    requested_lang: Optional[str],
    resolved_lang: str,
) -> Optional[str]:
    if requested_lang and requested_lang != resolved_lang:
        return requested_lang
    url = cast(Optional[str], state.get("user_response_language"))
    ql = cast(Optional[str], state.get("query_language"))
    target_pref = url or ql
    if target_pref and target_pref != resolved_lang:
        return target_pref
    return None


def _localize_book_name_for_target(
    canonical_book: str,
    target_code: str,
    *,
    resolve_bible_data_root: Callable[..., Any],
    load_book_titles: Callable[[Path], dict[str, str]],
    get_book_name: Callable[[str, str], str],
    translate_text: Callable[[str, str], str],
) -> str:
    try:
        t_root, _t_lang, _t_ver = resolve_bible_data_root(
            response_language=None,
            query_language=None,
            requested_lang=target_code,
            requested_version=None,
        )
        t_titles = load_book_titles(t_root)
        translated_book = t_titles.get(canonical_book)
    except FileNotFoundError:
        translated_book = None
    if not translated_book:
        static_name = get_book_name(target_code, canonical_book)
        if static_name != canonical_book:
            translated_book = static_name
        else:
            translated_book = translate_text(canonical_book, target_code)
    return cast(str, translated_book)


def _autotranslate_scripture_response(
    verses: list[tuple[str, str]],
    canonical_book: str,
    header_suffix: str,
    target_code: str,
    *,
    resolve_bible_data_root: Callable[..., Any],
    load_book_titles: Callable[[Path], dict[str, str]],
    get_book_name: Callable[[str, str], str],
    translate_text: Callable[[str, str], str],
) -> dict:
    translated_book = _localize_book_name_for_target(
        canonical_book,
        target_code,
        resolve_bible_data_root=resolve_bible_data_root,
        load_book_titles=load_book_titles,
        get_book_name=get_book_name,
        translate_text=translate_text,
    )
    translated_lines: list[str] = [
        _norm_ws(translate_text(str(txt), target_code)) for _ref, txt in verses
    ]
    translated_body = " ".join(translated_lines)
    return _make_scripture_response(
        content_language=target_code,
        header_book=translated_book,
        header_suffix=header_suffix,
        scripture_text=translated_body,
        header_is_translated=True,
    )


def retrieve_scripture(
    state: Any,
    *,
    resolve_selection_for_single_book: Callable[[str, str], tuple[str | None, list[tuple[int, int | None, int | None, int | None]] | None, str | None]],
    resolve_bible_data_root: Callable[..., Any],
    list_available_sources: Callable[[], list[tuple[str, str]]],
    load_book_titles: Callable[[Path], dict[str, str]],
    get_book_name: Callable[[str, str], str],
    select_verses: Callable[..., list[tuple[str, str]]],
    label_ranges: Callable[[str, Any], str],
    translate_text: Callable[[str, str], str],
    open_ai_client: OpenAI,
    add_tokens: Callable[..., None],
    extract_cached_input_tokens: Callable[[Any], int | None],
    ResponseLanguage: Any,
    Language: Any,
    TARGET_TRANSLATION_LANGUAGE_AGENT_SYSTEM_PROMPT: str,
    supported_language_map: dict[str, str],
) -> dict:
    """Handle retrieve-scripture with optional auto-translation."""
    s = cast(dict[str, Any], state)
    query = cast(str, s["transformed_query"])
    query_lang = cast(str, s["query_language"])
    logger.info("[retrieve-scripture] start; query_lang=%s; query=%s", query_lang, query)

    # 1) Parse passage selection
    canonical_book, ranges, err = resolve_selection_for_single_book(query, query_lang)
    if err:
        return {"responses": [{"intent": IntentType.RETRIEVE_SCRIPTURE, "response": err}]}
    assert canonical_book is not None and ranges is not None

    # 2) Detect explicit requested source language (e.g., "in Indonesian").
    requested_lang = _resolve_requested_source_language(
        query,
        open_ai_client=open_ai_client,
        add_tokens=add_tokens,
        extract_cached_input_tokens=extract_cached_input_tokens,
        ResponseLanguage=ResponseLanguage,
        Language=Language,
        TARGET_TRANSLATION_LANGUAGE_AGENT_SYSTEM_PROMPT=TARGET_TRANSLATION_LANGUAGE_AGENT_SYSTEM_PROMPT,
        supported_language_map=supported_language_map,
    )

    # 3) Resolve bible data root path with fallbacks
    src = _resolve_source_bible_with_requested(
        state=s,
        requested_lang=requested_lang,
        resolve_bible_data_root=resolve_bible_data_root,
        list_available_sources=list_available_sources,
    )
    if isinstance(src, str):
        return {"responses": [{"intent": IntentType.RETRIEVE_SCRIPTURE, "response": src}]}
    data_root, resolved_lang, _resolved_version = src

    # 4–5) Check limits and retrieve verses
    verses = _enforce_retrieve_limit_and_select(
        data_root=data_root,
        canonical_book=canonical_book,
        ranges=ranges,
        select_verses=select_verses,
        label_ranges=label_ranges,
    )
    if isinstance(verses, str):
        return {"responses": [{"intent": IntentType.RETRIEVE_SCRIPTURE, "response": verses}]}

    # 6) Header suffix and body paragraph
    suffix = _compute_header_suffix(label_ranges, canonical_book, ranges)
    scripture_text = " ".join(_norm_ws(str(txt)) for _ref, txt in verses)

    # 7) Decide on auto-translation target
    desired_target = _decide_retrieve_target(state=s, requested_lang=requested_lang, resolved_lang=cast(str, resolved_lang))

    # If we have a target and it's a supported language code, auto-translate body + header
    if desired_target and desired_target in supported_language_map:
        response_obj = _autotranslate_scripture_response(
            verses,
            canonical_book,
            suffix,
            desired_target,
            resolve_bible_data_root=resolve_bible_data_root,
            load_book_titles=load_book_titles,
            get_book_name=get_book_name,
            translate_text=translate_text,
        )
        return {"responses": [{"intent": IntentType.RETRIEVE_SCRIPTURE, "response": response_obj}]}

    # No auto-translation required; return verbatim with canonical header (to be localized downstream if desired)
    titles_map = load_book_titles(data_root)
    header_book = titles_map.get(canonical_book) or get_book_name(str(resolved_lang), canonical_book)
    response_obj = _make_scripture_response(
        content_language=str(resolved_lang),
        header_book=header_book,
        header_suffix=suffix,
        scripture_text=scripture_text,
        header_is_translated=False,
    )
    return {"responses": [{"intent": IntentType.RETRIEVE_SCRIPTURE, "response": response_obj}]}

