"""Intent handlers for passage-based operations (summary, keywords, retrieval, audio)."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, cast

from openai import OpenAIError
from openai.types.responses.easy_input_message_param import EasyInputMessageParam

from bt_servant_engine.core.config import config
from bt_servant_engine.core.intents import IntentType
from bt_servant_engine.core.language import (
    LANGUAGE_OTHER,
    ResponseLanguage,
    lookup_language_code,
    normalize_language_code,
)
from bt_servant_engine.core.logging import get_logger
from bt_servant_engine.services.cache_manager import CACHE_SCHEMA_VERSION, get_cache
from bt_servant_engine.services.openai_utils import track_openai_usage
from bt_servant_engine.services.passage_selection import (
    PassageSelectionRequest,
    resolve_selection_for_single_book,
)
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

SUMMARY_FOCUS_HINT = (
    "Focus only on the portion of the user's message that asked for a passage summary. "
    "Ignore any other requests or book references in the message."
)

MISSING_VERSES_MESSAGE = (
    "I couldn't locate those verses in the Bible data. Please check the reference and try again."
)

MISSING_KEYWORDS_MESSAGE = (
    "I couldn't locate keywords for that selection. Please check the reference and try again."
)


@dataclass(slots=True)
class PassageSelectionResult:
    """Normalized passage selection output with canonical book and ranges."""

    canonical_book: str
    ranges: list[RangeSelection]


# pylint: disable=too-many-instance-attributes
@dataclass(frozen=True)
class SummaryCacheKey:
    """Cache key for passage summaries."""

    schema: str
    canonical_book: str
    ranges: tuple["RangeSelection", ...]
    source_language: str
    source_version: str | None
    agentic_strength: str
    model_name: str
    verses_digest: str


@dataclass(frozen=True)
class KeywordsCacheKey:
    """Cache key for keyword lookups."""

    schema: str
    canonical_book: str
    ranges: tuple["RangeSelection", ...]
    data_root: str
    data_root_mtime: int


SUMMARY_CACHE_SCHEMA = f"{CACHE_SCHEMA_VERSION}:passage_summary:v1"
KEYWORDS_CACHE_SCHEMA = f"{CACHE_SCHEMA_VERSION}:passage_keywords:v1"

_SUMMARY_CACHE = get_cache("passage_summary")
_KEYWORDS_CACHE = get_cache("passage_keywords")


@dataclass(slots=True)
class PassageSummaryRequest:
    """Inputs required for generating a passage summary."""

    selection: PassageSelectionRequest
    user_response_language: Optional[str]
    agentic_strength: str
    model_for_agentic_strength: Callable[..., Any]
    extract_cached_input_tokens: Callable[..., Any]


@dataclass(slots=True)
class PassageSourceMetadata:
    """Resolved scripture source details for retrieval."""

    data_root: Path
    language: str
    version: str
    titles_map: dict[str, str]


@dataclass(slots=True)
class PassageKeywordsRequest:
    """Inputs required to compute passage keywords."""

    selection: PassageSelectionRequest


@dataclass(slots=True)
class RetrieveScriptureRequest:
    """Inputs required to retrieve scripture text for a passage."""

    selection: PassageSelectionRequest
    user_response_language: Optional[str]
    agentic_strength: str
    extract_cached_input_tokens: Callable[..., Any]


@dataclass(slots=True)
class RetrievedPassage:
    """Retrieved passage data including localized reference and verse text."""

    selection: PassageSelectionResult
    source: PassageSourceMetadata
    verses: list[tuple[str, str]]
    ref_label: str
    suffix: str
    scripture_text: str


@dataclass(slots=True)
class ListenToScriptureRequest:
    """Inputs required to produce an audio-friendly scripture response."""

    retrieve_request: RetrieveScriptureRequest
    reconstruct_structured_text: Callable[..., Any]


def _resolve_passage_selection(
    request: PassageSelectionRequest,
) -> tuple[Optional[PassageSelectionResult], Optional[str]]:
    canonical_book, ranges, err = resolve_selection_for_single_book(request)
    if err:
        return None, err
    assert canonical_book is not None and ranges is not None  # nosec B101
    normalized_ranges = [cast(RangeSelection, tuple(r)) for r in ranges]
    return PassageSelectionResult(canonical_book=canonical_book, ranges=normalized_ranges), None


def _passage_followup_context(
    intent: IntentType, selection: PassageSelectionResult
) -> dict[str, Any]:
    return {
        "intent": intent,
        "book": selection.canonical_book,
        "ranges": list(selection.ranges),
    }


def _resolve_bible_source(
    user_response_language: Optional[str],
    query_language: str,
    *,
    requested_lang: Optional[str] = None,
) -> tuple[Optional[PassageSourceMetadata], Optional[str]]:
    try:
        data_root, resolved_lang, resolved_version = resolve_bible_data_root(
            response_language=user_response_language,
            query_language=query_language,
            requested_lang=requested_lang,
            requested_version=None,
        )
        titles_map = load_book_titles(data_root)
        return (
            PassageSourceMetadata(
                data_root=data_root,
                language=str(resolved_lang),
                version=resolved_version,
                titles_map=titles_map,
            ),
            None,
        )
    except FileNotFoundError:
        message = (
            "Scripture data is not available on this server. Please contact the administrator."
        )
        return None, message


def _localize_reference(
    selection: PassageSelectionResult,
    source: PassageSourceMetadata,
) -> tuple[str, str]:
    localized_book = source.titles_map.get(selection.canonical_book) or get_book_name(
        source.language,
        selection.canonical_book,
    )
    ref_label_en = label_ranges(selection.canonical_book, selection.ranges)
    if ref_label_en == selection.canonical_book:
        return localized_book, localized_book
    if ref_label_en.startswith(f"{selection.canonical_book} "):
        suffix = ref_label_en[len(selection.canonical_book) + 1 :]
        return localized_book, f"{localized_book} {suffix}"
    return localized_book, ref_label_en


def _select_passage_verses(
    selection: PassageSelectionResult,
    source: PassageSourceMetadata,
) -> list[tuple[str, str]]:
    verses = select_verses(source.data_root, selection.canonical_book, selection.ranges)
    logger.info(
        "[passage] retrieved %d verse(s) for book=%s",
        len(verses),
        selection.canonical_book,
    )
    return verses


def _join_passage_text(verses: Iterable[tuple[str, str]]) -> str:
    return "\n".join(f"{ref}: {txt}" for ref, txt in verses)


def _build_summary_messages(ref_label: str, verses_text: str) -> list[EasyInputMessageParam]:
    return [
        {
            "role": "developer",
            "content": SUMMARY_FOCUS_HINT,
        },
        {"role": "developer", "content": f"Passage reference: {ref_label}"},
        {"role": "developer", "content": f"Passage verses (use only this content):\n{verses_text}"},
        {"role": "user", "content": "Provide a concise, faithful summary of the passage above."},
    ]


def _summarize_passage(
    ref_label: str,
    verses_text: str,
    verses_count: int,
    request: PassageSummaryRequest,
    model_name: str,
) -> str:
    messages = _build_summary_messages(ref_label, verses_text)
    deps = request.selection.dependencies
    logger.info("[passage-summary] summarizing %d verses", verses_count)
    summary_resp = deps.client.responses.create(
        model=model_name,
        instructions=PASSAGE_SUMMARY_AGENT_SYSTEM_PROMPT,
        input=cast(Any, messages),
        store=False,
    )
    usage = getattr(summary_resp, "usage", None)
    track_openai_usage(usage, model_name, request.extract_cached_input_tokens, add_tokens)
    summary_text = summary_resp.output_text
    logger.info(
        "[passage-summary] summary generated (len=%d)",
        len(summary_text) if summary_text else 0,
    )
    return summary_text


def _normalize_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _suffix_from_label(canonical_book: str, ref_label: str) -> str:
    if ref_label == canonical_book:
        return ""
    if ref_label.startswith(f"{canonical_book} "):
        return ref_label[len(canonical_book) + 1 :]
    return ref_label


def _assemble_scripture_text(verses: Iterable[tuple[str, str]]) -> str:
    return " ".join(_normalize_whitespace(str(txt)) for _ref, txt in verses)


def _resolve_translated_book_name(
    canonical_book: str,
    desired_target: str,
    agentic_strength: str,
    translate_text_fn: Callable[..., Any],
) -> str:
    try:
        t_root, _t_lang, _t_ver = resolve_bible_data_root(
            response_language=None,
            query_language=None,
            requested_lang=desired_target,
            requested_version=None,
        )
        t_titles = load_book_titles(t_root)
        translated = t_titles.get(canonical_book)
        if translated:
            return translated
    except FileNotFoundError:
        translated = None
    static_name = get_book_name(desired_target, canonical_book)
    if static_name != canonical_book:
        return static_name
    return str(
        translate_text_fn(
            response_text=canonical_book,
            target_language=desired_target,
            agentic_strength=agentic_strength,
        )
    )


def _detect_requested_language(request: RetrieveScriptureRequest) -> Optional[str]:
    deps = request.selection.dependencies
    query = request.selection.query
    logger.info("[retrieve-scripture] detecting requested language for query")
    try:
        tl_resp = deps.client.responses.parse(
            model="gpt-4o",
            instructions=TARGET_TRANSLATION_LANGUAGE_AGENT_SYSTEM_PROMPT,
            input=cast(Any, [{"role": "user", "content": f"message: {query}"}]),
            text_format=ResponseLanguage,
            temperature=0,
            store=False,
        )
        tl_usage = getattr(tl_resp, "usage", None)
        track_openai_usage(
            tl_usage,
            "gpt-4o",
            request.extract_cached_input_tokens,
            add_tokens,
        )
        tl_parsed = cast(ResponseLanguage | None, tl_resp.output_parsed)
        if tl_parsed and tl_parsed.language != LANGUAGE_OTHER:
            return str(tl_parsed.language)
    except OpenAIError:
        logger.info(
            "[retrieve-scripture] requested-language parse failed; will fallback",
            exc_info=True,
        )
    except (ValueError, TypeError, KeyError):
        logger.info(
            "[retrieve-scripture] requested-language parse failed (generic); will fallback",
            exc_info=True,
        )
    match = re.search(
        r"\b(?:in|from the|from)\s+([A-Za-z][A-Za-z\- ]{1,30})\b",
        query,
        flags=re.IGNORECASE,
    )
    if match:
        name = match.group(1).strip().title()
        lookup = lookup_language_code(name)
        if lookup:
            return lookup
    return None


def _resolve_retrieval_source(
    request: RetrieveScriptureRequest,
    requested_lang: Optional[str],
) -> tuple[Optional[PassageSourceMetadata], Optional[str]]:
    try:
        data_root, resolved_lang, resolved_version = resolve_bible_data_root(
            response_language=request.user_response_language,
            query_language=request.selection.query_lang,
            requested_lang=requested_lang,
            requested_version=None,
        )
        titles_map = load_book_titles(data_root)
        return (
            PassageSourceMetadata(
                data_root=data_root,
                language=str(resolved_lang),
                version=resolved_version,
                titles_map=titles_map,
            ),
            None,
        )
    except FileNotFoundError:
        available = list_available_sources()
        if not available:
            message = (
                "Scripture data is not available on this server. Please contact the administrator."
            )
            return None, message
        options = ", ".join(f"{lang}/{ver}" for lang, ver in available)
        message = (
            f"I couldn't find a Bible source matching your request. Available sources: {options}. "
            "Would you like me to use one of these?"
        )
        return None, message


def _retrieve_passage_content(
    selection: PassageSelectionResult,
    source: PassageSourceMetadata,
) -> tuple[Optional[RetrievedPassage], Optional[str]]:
    verses = select_verses(source.data_root, selection.canonical_book, selection.ranges)
    if not verses:
        return None, MISSING_VERSES_MESSAGE
    if len(verses) > config.RETRIEVE_SCRIPTURE_VERSE_LIMIT:
        ref_label = label_ranges(selection.canonical_book, selection.ranges)
        message = (
            f"I can only retrieve up to {config.RETRIEVE_SCRIPTURE_VERSE_LIMIT} verses at a time. "
            f"Your selection {ref_label} includes {len(verses)} verses. "
            "Please narrow the range (e.g., a chapter or a shorter span)."
        )
        return None, message
    ref_label = label_ranges(selection.canonical_book, selection.ranges)
    suffix = _suffix_from_label(selection.canonical_book, ref_label)
    scripture_text = _assemble_scripture_text(verses)
    return (
        RetrievedPassage(
            selection=selection,
            source=source,
            verses=verses,
            ref_label=ref_label,
            suffix=suffix,
            scripture_text=scripture_text,
        ),
        None,
    )


def _determine_desired_target(
    resolved_language: str,
    requested_lang: Optional[str],
    request: RetrieveScriptureRequest,
) -> Optional[str]:
    if requested_lang:
        normalized_request = normalize_language_code(requested_lang)
        if normalized_request and normalized_request != resolved_language:
            return normalized_request
        return None
    preferred = normalize_language_code(request.user_response_language) or normalize_language_code(
        request.selection.query_lang
    )
    if preferred and preferred != resolved_language:
        return preferred
    return None


def _auto_translate_passage(
    passage: RetrievedPassage,
    desired_target: str,
    request: RetrieveScriptureRequest,
) -> dict[str, Any]:
    translated_book = _resolve_translated_book_name(
        passage.selection.canonical_book,
        desired_target,
        request.agentic_strength,
        request.selection.dependencies.translate_text,
    )
    translated_lines = [
        _normalize_whitespace(
            request.selection.dependencies.translate_text(
                response_text=str(txt),
                target_language=desired_target,
                agentic_strength=request.agentic_strength,
            )
        )
        for _ref, txt in passage.verses
    ]
    translated_body = " ".join(translated_lines)
    return {
        "suppress_translation": True,
        "content_language": desired_target,
        "header_is_translated": True,
        "segments": [
            {"type": "header_book", "text": translated_book},
            {"type": "header_suffix", "text": passage.suffix},
            {"type": "scripture", "text": translated_body},
        ],
    }


def _build_verbatim_passage_response(passage: RetrievedPassage) -> dict[str, Any]:
    header_book = passage.source.titles_map.get(passage.selection.canonical_book) or get_book_name(
        passage.source.language,
        passage.selection.canonical_book,
    )
    return {
        "suppress_translation": True,
        "content_language": passage.source.language,
        "header_is_translated": False,
        "segments": [
            {"type": "header_book", "text": header_book},
            {"type": "header_suffix", "text": passage.suffix},
            {"type": "scripture", "text": passage.scripture_text},
        ],
    }


def get_passage_summary(request: PassageSummaryRequest) -> dict[str, Any]:  # pylint: disable=too-many-locals
    """Handle get-passage-summary: extract refs, retrieve verses, summarize.

    - If user query language is not English, translate the transformed query to English
      for extraction only.
    - Extract passage selection via structured LLM parse with a strict prompt and
      canonical book list.
    - Validate constraints (single book, up to whole book; no cross-book).
    - Load verses from sources/bible_data/en efficiently and summarize.
    - Return a single combined summary prefixed with a canonical reference echo.
    """
    deps = request.selection.dependencies
    logger.info(
        "[passage-summary] start; query_lang=%s; query=%s",
        request.selection.query_lang,
        request.selection.query,
    )

    selection_result, err = _resolve_passage_selection(
        PassageSelectionRequest(
            query=request.selection.query,
            query_lang=request.selection.query_lang,
            dependencies=deps,
            focus_hint=request.selection.focus_hint or SUMMARY_FOCUS_HINT,
        )
    )
    if err:
        return {"responses": [{"intent": IntentType.GET_PASSAGE_SUMMARY, "response": err}]}
    assert selection_result is not None  # nosec B101

    source, source_error = _resolve_bible_source(
        request.user_response_language,
        request.selection.query_lang,
    )
    if source_error:
        return {"responses": [{"intent": IntentType.GET_PASSAGE_SUMMARY, "response": source_error}]}
    assert source is not None  # nosec B101
    logger.info(
        "[passage-summary] retrieving verses from %s (lang=%s, version=%s)",
        source.data_root,
        source.language,
        source.version,
    )

    verses = _select_passage_verses(selection_result, source)
    if not verses:
        logger.info("[passage-summary] no verses found for selection; prompting user")
        return {
            "responses": [
                {"intent": IntentType.GET_PASSAGE_SUMMARY, "response": MISSING_VERSES_MESSAGE}
            ]
        }

    _, ref_label = _localize_reference(selection_result, source)
    logger.info("[passage-summary] label=%s", ref_label)
    verses_text = _join_passage_text(verses)
    model_name = request.model_for_agentic_strength(
        request.agentic_strength,
        allow_low=True,
        allow_very_low=True,
    )
    ranges_tuple = tuple(selection_result.ranges)
    verses_digest = hashlib.sha256(verses_text.encode("utf-8")).hexdigest()
    cache_key = SummaryCacheKey(
        schema=SUMMARY_CACHE_SCHEMA,
        canonical_book=selection_result.canonical_book,
        ranges=ranges_tuple,
        source_language=source.language,
        source_version=source.version,
        agentic_strength=request.agentic_strength,
        model_name=model_name,
        verses_digest=verses_digest,
    )

    def _compute_summary() -> dict[str, Any]:
        summary_text = _summarize_passage(
            ref_label,
            verses_text,
            len(verses),
            request,
            model_name,
        )
        response_text = f"Summary of {ref_label}:\n\n{summary_text}"
        logger.info("[passage-summary] done (generated)")
        return {
            "responses": [
                {
                    "intent": IntentType.GET_PASSAGE_SUMMARY,
                    "response": response_text,
                }
            ],
            "passage_followup_context": _passage_followup_context(
                IntentType.GET_PASSAGE_SUMMARY,
                selection_result,
            ),
        }

    summary_response, hit = _SUMMARY_CACHE.get_or_set(cache_key, _compute_summary)
    if hit:
        logger.info("[passage-summary] served from cache label=%s", ref_label)
    return summary_response


def get_passage_keywords(request: PassageKeywordsRequest) -> dict[str, Any]:
    """Handle get-passage-keywords: extract refs, retrieve keywords, and list them.

    Mirrors the summary flow for selection parsing and validation, but instead of
    summarizing verses, loads per-verse keyword data from sources/keyword_data and
    returns a comma-separated list of distinct tw_match values present in the
    selection. The response is prefixed with "Keywords in <range>\n\n".
    """
    logger.info(
        "[passage-keywords] start; query_lang=%s; query=%s",
        request.selection.query_lang,
        request.selection.query,
    )
    selection_result, err = _resolve_passage_selection(request.selection)
    if err:
        return {"responses": [{"intent": IntentType.GET_PASSAGE_KEYWORDS, "response": err}]}
    assert selection_result is not None  # nosec B101

    # Retrieve keywords from keyword dataset
    data_root = Path("sources") / "keyword_data"
    try:
        data_root_mtime = int(data_root.stat().st_mtime_ns)
    except OSError:
        data_root_mtime = 0
    cache_key = KeywordsCacheKey(
        schema=KEYWORDS_CACHE_SCHEMA,
        canonical_book=selection_result.canonical_book,
        ranges=tuple(selection_result.ranges),
        data_root=str(data_root),
        data_root_mtime=data_root_mtime,
    )

    def _compute_keywords() -> dict[str, Any]:
        logger.info("[passage-keywords] retrieving keywords from %s", data_root)
        keywords = select_keywords(
            data_root,
            selection_result.canonical_book,
            selection_result.ranges,
        )
        logger.info("[passage-keywords] retrieved %d keyword(s)", len(keywords))

        if not keywords:
            logger.info("[passage-keywords] no keywords found; prompting user")
            return {
                "responses": [
                    {
                        "intent": IntentType.GET_PASSAGE_KEYWORDS,
                        "response": MISSING_KEYWORDS_MESSAGE,
                    }
                ],
            }

        ref_label = label_ranges(selection_result.canonical_book, selection_result.ranges)
        header = f"Keywords in {ref_label}\n\n"
        body = ", ".join(keywords)
        response_text = header + body
        logger.info("[passage-keywords] done (generated)")
        return {
            "responses": [
                {
                    "intent": IntentType.GET_PASSAGE_KEYWORDS,
                    "response": response_text,
                }
            ],
            "passage_followup_context": _passage_followup_context(
                IntentType.GET_PASSAGE_KEYWORDS,
                selection_result,
            ),
        }

    keyword_response, hit = _KEYWORDS_CACHE.get_or_set(cache_key, _compute_keywords)
    if hit:
        logger.info(
            "[passage-keywords] served from cache for book=%s",
            selection_result.canonical_book,
        )
    return keyword_response


def retrieve_scripture(request: RetrieveScriptureRequest) -> dict[str, Any]:
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
    logger.info(
        "[retrieve-scripture] start; query_lang=%s; query=%s",
        request.selection.query_lang,
        request.selection.query,
    )

    selection_result, err = _resolve_passage_selection(request.selection)
    if err:
        return {"responses": [{"intent": IntentType.RETRIEVE_SCRIPTURE, "response": err}]}
    assert selection_result is not None  # nosec B101

    requested_lang = _detect_requested_language(request)
    source, source_error = _resolve_retrieval_source(request, requested_lang)
    if source_error:
        return {"responses": [{"intent": IntentType.RETRIEVE_SCRIPTURE, "response": source_error}]}
    assert source is not None  # nosec B101
    logger.info(
        "[retrieve-scripture] data_root=%s lang=%s version=%s",
        source.data_root,
        source.language,
        source.version,
    )

    passage, retrieval_error = _retrieve_passage_content(selection_result, source)
    if retrieval_error:
        return {
            "responses": [{"intent": IntentType.RETRIEVE_SCRIPTURE, "response": retrieval_error}]
        }
    assert passage is not None  # nosec B101

    desired_target = _determine_desired_target(
        source.language,
        requested_lang,
        request,
    )
    if desired_target:
        logger.info(
            "[retrieve-scripture] auto-translating scripture to %s",
            desired_target,
        )
        response_obj = _auto_translate_passage(passage, desired_target, request)
    else:
        response_obj = _build_verbatim_passage_response(passage)

    return {
        "responses": [{"intent": IntentType.RETRIEVE_SCRIPTURE, "response": response_obj}],
        "passage_followup_context": _passage_followup_context(
            IntentType.RETRIEVE_SCRIPTURE,
            selection_result,
        ),
    }


def listen_to_scripture(request: ListenToScriptureRequest) -> dict[str, Any]:
    """Delegate to retrieve-scripture and request voice delivery.

    Reuses retrieve-scripture end-to-end (selection, retrieval, translation, formatting)
    and sets a delivery hint that the API should send a voice message.
    """
    out = retrieve_scripture(request.retrieve_request)
    out["send_voice_message"] = True
    responses = cast(list[dict], out.get("responses", []))
    if responses:
        # Reconstruct scripture text for voice playback using the structured response.
        out["voice_message_text"] = request.reconstruct_structured_text(
            resp_item=responses[0],
            localize_to=None,
        )
        for resp in responses:
            resp["suppress_text_delivery"] = True
    ctx = out.get("passage_followup_context")
    if isinstance(ctx, dict):
        ctx["intent"] = IntentType.LISTEN_TO_SCRIPTURE
    return out


__all__ = [
    "PASSAGE_SUMMARY_AGENT_SYSTEM_PROMPT",
    "TARGET_TRANSLATION_LANGUAGE_AGENT_SYSTEM_PROMPT",
    "get_passage_summary",
    "get_passage_keywords",
    "retrieve_scripture",
    "listen_to_scripture",
]
RangeSelection = tuple[int, int | None, int | None, int | None]
