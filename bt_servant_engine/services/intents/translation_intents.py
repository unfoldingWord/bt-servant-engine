"""Intent handlers for scripture translation and translation helps."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, cast

from openai import OpenAI, OpenAIError
from openai.types.responses.easy_input_message_param import EasyInputMessageParam

from bt_servant_engine.core.config import config
from bt_servant_engine.core.intents import IntentType
from bt_servant_engine.core.language import (
    LANGUAGE_OTHER,
    ResponseLanguage,
    TranslatedPassage,
    friendly_language_name,
    lookup_language_code,
    normalize_language_code,
)
from bt_servant_engine.core.logging import get_logger
from bt_servant_engine.services.cache_manager import CACHE_SCHEMA_VERSION, get_cache
from bt_servant_engine.services.openai_utils import track_openai_usage
from bt_servant_engine.services.passage_selection import (
    PassageSelectionDependencies,
    PassageSelectionRequest,
    resolve_selection_for_single_book,
)
from bt_servant_engine.services.intents.passage_intents import (
    TARGET_TRANSLATION_LANGUAGE_AGENT_SYSTEM_PROMPT,
)
from bt_servant_engine.services.translation_helpers import (
    TranslationHelpsRequest,
    TranslationRange,
)
from utils.bible_data import list_available_sources, resolve_bible_data_root
from utils.bsb import label_ranges, select_verses
from utils.perf import add_tokens

logger = get_logger(__name__)

LANGUAGE_REGEX = re.compile(
    r"\b(?:into|to|in)\s+([A-Za-z][A-Za-z\- ]{1,30})\b", flags=re.IGNORECASE
)

RangeSelection = tuple[int, int | None, int | None, int | None]

TRANSLATE_FOCUS_HINT = (
    "Focus only on the portion of the user's message that asked to translate scripture. "
    "Ignore any other requests or book references in the message."
)

TRANSLATION_HELPS_FOCUS_HINT = (
    "Focus only on the portion of the user's message that asked for translation helps. "
    "Ignore any other requests or book references in the message."
)


@dataclass(slots=True)
class TranslationContext:
    """Normalized translation metadata and verse payload."""

    canonical_book: str
    ranges: list[RangeSelection]
    target_code: str
    resolved_lang: str
    body_source: str
    header_suffix: str
    verses: list[tuple[str, str]]


@dataclass(slots=True)
class TranslationSourceMetadata:
    """Location and language metadata for scripture retrieval."""

    data_root: Path
    language: str
    version: str


@dataclass(slots=True)
class TranslationTruncationDetails:
    """Metadata describing a truncated translation helps selection."""

    original_label: Optional[str]


@dataclass(slots=True)
class TranslationHelpsPayload:
    """Intermediate structure for translation helps generation."""

    canonical_book: str
    ranges: list[RangeSelection]
    ref_label: str
    context_obj: dict[str, Any]
    raw_helps: list[Any]
    truncation: Optional[TranslationTruncationDetails]


@dataclass(slots=True)
class TranslationRequestParams:
    """Inputs required to translate scripture for the user."""

    client: OpenAI
    query: str
    query_lang: str
    book_map: dict[str, Any]
    user_response_language: Optional[str]
    agentic_strength: str


@dataclass(slots=True)
class TranslationDependencies:
    """External helpers used while resolving translation context."""

    detect_books_fn: Callable[..., Any]
    translate_text_fn: Callable[..., Any]
    select_model_fn: Callable[..., Any]
    extract_cached_tokens_fn: Callable[..., Any]


class TranslationContextError(Exception):
    """Exception carrying a pre-built intent response."""

    def __init__(self, response: dict[str, Any]):
        super().__init__("translation context unavailable")
        self.response = response


@dataclass(slots=True)
class TranslationHelpsRequestParams:
    """Inputs required to gather translation helps for a selection."""

    client: OpenAI
    query: str
    query_lang: str
    book_map: dict[str, Any]
    agentic_strength: str
    user_response_language: Optional[str] = None


@dataclass(slots=True)
class TranslationHelpsDependencies:
    """Helpers needed to build translation helps guidance."""

    detect_books_fn: Callable[..., Any]
    translate_text_fn: Callable[..., Any]
    select_model_fn: Callable[..., Any]
    extract_cached_tokens_fn: Callable[..., Any]
    prepare_translation_helps_fn: Callable[..., Any]
    build_context_fn: Callable[..., Any]
    build_messages_fn: Callable[..., Any]


@dataclass(frozen=True)
class TranslationHelpsCacheKey:
    """Cache key for translation helps responses."""

    schema: str
    canonical_book: str
    ranges: tuple[TranslationRange, ...]
    agentic_strength: str
    model_name: str
    raw_helps_digest: str


TRANSLATION_HELPS_CACHE_SCHEMA = f"{CACHE_SCHEMA_VERSION}:translation_helps:v1"
_TRANSLATION_HELPS_CACHE = get_cache("translation_helps")


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


def _simple_response(message: str) -> dict[str, Any]:
    return {"responses": [{"intent": IntentType.TRANSLATE_SCRIPTURE, "response": message}]}


def _language_guidance_response(requested_name: Optional[str]) -> dict[str, Any]:
    label = requested_name or "that language"
    guidance = (
        f"I couldn't determine how to translate into {label}.\n\n"
        "Please mention the language explicitly or provide its ISO 639-1 code (for example: "
        "'tr' for Turkish, 'yo' for Yoruba, 'id' for Indonesian) so I can translate "
        "Scripture accordingly."
    )
    return _simple_response(guidance)


def _translation_helps_response(message: str) -> dict[str, Any]:
    return {"responses": [{"intent": IntentType.GET_TRANSLATION_HELPS, "response": message}]}


MISSING_VERSES_RESPONSE = _simple_response(
    "I couldn't locate those verses in the Bible data. Please check the reference and try again."
)


def _structured_target_language(
    client: OpenAI,
    query: str,
    extract_cached_input_tokens_fn: Callable[..., Any],
) -> Optional[str]:
    try:
        tl_resp = client.responses.parse(
            model="gpt-4o",
            instructions=TARGET_TRANSLATION_LANGUAGE_AGENT_SYSTEM_PROMPT,
            input=cast(Any, [{"role": "user", "content": f"message: {query}"}]),
            text_format=ResponseLanguage,
            temperature=0,
            store=False,
        )
    except OpenAIError:
        logger.info(
            "[translate-scripture] target-language parse failed; will fallback", exc_info=True
        )
        return None
    except (ValueError, TypeError) as exc:
        logger.warning(
            "[translate-scripture] target-language parse raised %s; will fallback",
            exc.__class__.__name__,
        )
        return None

    tl_usage = getattr(tl_resp, "usage", None)
    track_openai_usage(tl_usage, "gpt-4o", extract_cached_input_tokens_fn, add_tokens)
    tl_parsed = cast(ResponseLanguage | None, tl_resp.output_parsed)
    if tl_parsed and tl_parsed.language != LANGUAGE_OTHER:
        return str(tl_parsed.language)
    return None


def _extract_explicit_language(query: str) -> Optional[str]:
    match = LANGUAGE_REGEX.search(query)
    if match:
        return match.group(1).strip().title()
    return None


def _resolve_target_language(
    request: TranslationRequestParams,
    dependencies: TranslationDependencies,
) -> tuple[Optional[str], Optional[str]]:
    structured_code = _structured_target_language(
        request.client, request.query, dependencies.extract_cached_tokens_fn
    )
    explicit_name = _extract_explicit_language(request.query) if structured_code is None else None

    if structured_code:
        normalized = normalize_language_code(structured_code)
        if normalized and normalized != LANGUAGE_OTHER:
            return normalized, None

    if explicit_name:
        lookup = lookup_language_code(explicit_name)
        if lookup:
            return lookup, None
        return None, explicit_name

    preferred = normalize_language_code(request.user_response_language)
    if preferred and preferred != LANGUAGE_OTHER:
        return preferred, None

    detected = normalize_language_code(request.query_lang)
    if detected and detected != LANGUAGE_OTHER:
        return detected, None

    return None, None


def _build_header_suffix(
    canonical_book: str, ranges: list[tuple[int, int | None, int | None, int | None]]
) -> str:
    ref_label = label_ranges(canonical_book, ranges)
    if ref_label == canonical_book:
        return ""
    if ref_label.startswith(f"{canonical_book} "):
        return ref_label[len(canonical_book) + 1 :]
    return ref_label


def _normalize_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _compose_body_text(verses: list[tuple[str, str]]) -> str:
    return " ".join(_normalize_whitespace(text) for _, text in verses)


def _make_selection_dependencies(
    request: TranslationRequestParams,
    dependencies: TranslationDependencies,
) -> PassageSelectionDependencies:
    return PassageSelectionDependencies(
        client=request.client,
        book_map=request.book_map,
        detect_mentioned_books=cast(Callable[[str], list[str]], dependencies.detect_books_fn),
        translate_text=cast(Callable[[str, str], str], dependencies.translate_text_fn),
    )


def _resolve_translation_selection(
    request: TranslationRequestParams,
    dependencies: TranslationDependencies,
) -> tuple[str, list[RangeSelection]]:
    selection_dependencies = _make_selection_dependencies(request, dependencies)
    selection_request = PassageSelectionRequest(
        query=request.query,
        query_lang=request.query_lang,
        dependencies=selection_dependencies,
        focus_hint=TRANSLATE_FOCUS_HINT,
    )
    canonical_book, ranges, err = resolve_selection_for_single_book(selection_request)
    if err:
        raise TranslationContextError(_simple_response(err))
    if canonical_book is None or ranges is None:
        raise TranslationContextError(_simple_response("I couldn't identify those verses."))
    normalized_ranges = [cast(RangeSelection, tuple(r)) for r in ranges]
    return canonical_book, normalized_ranges


def _determine_target_code(
    request: TranslationRequestParams,
    dependencies: TranslationDependencies,
) -> str:
    target_code, requested_name = _resolve_target_language(request, dependencies)
    normalized = normalize_language_code(target_code)
    if normalized and normalized != LANGUAGE_OTHER:
        return normalized

    name = requested_name or friendly_language_name(target_code, fallback="that language")
    raise TranslationContextError(_language_guidance_response(name))


def _load_translation_source(request: TranslationRequestParams) -> TranslationSourceMetadata:
    try:
        data_root, resolved_lang, resolved_version = resolve_bible_data_root(
            response_language=request.user_response_language,
            query_language=request.query_lang,
            requested_lang=None,
            requested_version=None,
        )
        logger.info(
            "[translate-scripture] source data_root=%s lang=%s version=%s",
            data_root,
            resolved_lang,
            resolved_version,
        )
        return TranslationSourceMetadata(
            data_root=data_root,
            language=str(resolved_lang),
            version=resolved_version,
        )
    except FileNotFoundError as exc:
        available_sources = list_available_sources()
        if available_sources:
            available = "\n".join(f"- {src}" for src in available_sources)
        else:
            available = "None"
        message = (
            "The requested scripture source is unavailable right now.\n\n"
            "Available sources include:\n"
            f"{available}"
        )
        raise TranslationContextError(_simple_response(message)) from exc


def _load_translation_verses(
    data_root: Path,
    canonical_book: str,
    ranges: list[tuple[int, int | None, int | None, int | None]],
) -> list[tuple[str, str]]:
    try:
        verses = select_verses(data_root, canonical_book, ranges)
    except (RuntimeError, ValueError) as exc:
        logger.error("[translate-scripture] Failed to load verses: %s", exc, exc_info=True)
        raise TranslationContextError(
            _simple_response("I couldn't load those verses. Please try a different passage.")
        ) from exc

    if not verses:
        raise TranslationContextError(MISSING_VERSES_RESPONSE)
    return verses


def _maybe_collect_truncation_details(
    canonical_book: str,
    ref_label: str,
    metadata: Optional[dict[str, Any]],
    context_obj: dict[str, Any],
) -> Optional[TranslationTruncationDetails]:
    if not metadata or not metadata.get("truncated"):
        return None

    original_ranges = cast(Optional[list[TranslationRange]], metadata.get("original_ranges"))
    original_label_value = metadata.get("original_label")
    if original_label_value is None and original_ranges is not None:
        original_label_value = label_ranges(canonical_book, original_ranges)
    original_label = cast(Optional[str], original_label_value)

    selection_section = context_obj.setdefault("selection", {})
    selection_section["truncated"] = True
    selection_section["delivered_label"] = ref_label
    if original_label:
        selection_section["truncated_from"] = original_label
    return TranslationTruncationDetails(original_label=original_label)


def _make_translation_helps_request(
    request: TranslationHelpsRequestParams,
    dependencies: TranslationHelpsDependencies,
) -> TranslationHelpsRequest:
    # Select language-specific translation helps directory
    lang = request.user_response_language or "en"
    th_root = Path("sources") / "translation_helps"  # default English

    if lang != "en":
        localized_root = Path("sources") / f"translation_helps_{lang}"
        if localized_root.exists():
            th_root = localized_root
            logger.info(
                "[translation-helps] using localized helps for %s from %s",
                lang,
                th_root,
            )
        else:
            logger.warning(
                "[translation-helps] no localized helps for %s; falling back to English",
                lang,
            )
    else:
        logger.info("[translation-helps] loading helps from %s", th_root)

    bsb_root = Path("sources") / "bible_data" / "en" / "bsb"

    selection_dependencies = PassageSelectionDependencies(
        client=request.client,
        book_map=request.book_map,
        detect_mentioned_books=cast(Callable[[str], list[str]], dependencies.detect_books_fn),
        translate_text=cast(Callable[[str, str], str], dependencies.translate_text_fn),
    )

    return TranslationHelpsRequest(
        query=request.query,
        query_lang=request.query_lang,
        th_root=th_root,
        bsb_root=bsb_root,
        dependencies=selection_dependencies,
        selection_focus_hint=TRANSLATION_HELPS_FOCUS_HINT,
    )


def _prepare_translation_helps_payload(
    request: TranslationHelpsRequestParams,
    dependencies: TranslationHelpsDependencies,
) -> tuple[Optional[TranslationHelpsPayload], Optional[dict[str, Any]]]:
    translation_request = _make_translation_helps_request(request, dependencies)

    canonical_book, ranges, raw_helps, metadata, err = dependencies.prepare_translation_helps_fn(
        translation_request
    )
    if err:
        return None, _translation_helps_response(err)
    if canonical_book is None or ranges is None or raw_helps is None:
        return None, _translation_helps_response(
            "I couldn't locate translation helps for that selection. Please try another reference."
        )

    normalized_ranges = [tuple(r) for r in ranges]
    original_ranges = (
        cast(Optional[list[TranslationRange]], metadata.get("original_ranges"))
        if metadata and metadata.get("original_ranges")
        else None
    )
    ref_label, context_obj = dependencies.build_context_fn(
        canonical_book,
        normalized_ranges,
        raw_helps,
        original_ranges=original_ranges,
    )
    truncation = _maybe_collect_truncation_details(canonical_book, ref_label, metadata, context_obj)
    payload = TranslationHelpsPayload(
        canonical_book=canonical_book,
        ranges=normalized_ranges,
        ref_label=ref_label,
        context_obj=context_obj,
        raw_helps=raw_helps,
        truncation=truncation,
    )
    return payload, None


def _build_translation_context(
    request: TranslationRequestParams,
    dependencies: TranslationDependencies,
) -> TranslationContext:
    canonical_book, ranges = _resolve_translation_selection(request, dependencies)
    target_code = _determine_target_code(request, dependencies)
    source_metadata = _load_translation_source(request)
    verses = _load_translation_verses(source_metadata.data_root, canonical_book, ranges)
    return TranslationContext(
        canonical_book=canonical_book,
        ranges=ranges,
        target_code=target_code,
        resolved_lang=source_metadata.language,
        body_source=_compose_body_text(verses),
        header_suffix=_build_header_suffix(canonical_book, ranges),
        verses=verses,
    )


def _attempt_structured_translation(
    request: TranslationRequestParams,
    context: TranslationContext,
    dependencies: TranslationDependencies,
) -> Optional[TranslatedPassage]:
    messages: list[EasyInputMessageParam] = [
        {"role": "developer", "content": f"canonical_book: {context.canonical_book}"},
        {
            "role": "developer",
            "content": f"header_suffix (do not translate): {context.header_suffix}",
        },
        {"role": "developer", "content": f"target_language: {context.target_code}"},
        {"role": "developer", "content": "passage body (translate; preserve newlines):"},
        {"role": "developer", "content": context.body_source},
    ]
    model_name = dependencies.select_model_fn(
        request.agentic_strength, allow_low=False, allow_very_low=True
    )
    try:
        resp = request.client.responses.parse(
            model=model_name,
            instructions=TRANSLATE_PASSAGE_AGENT_SYSTEM_PROMPT,
            input=cast(Any, messages),
            text_format=TranslatedPassage,
            temperature=0,
            store=False,
        )
    except OpenAIError:
        logger.warning(
            "[translate-scripture] structured parse failed due to OpenAI error; falling back.",
            exc_info=True,
        )
        return None
    except (ValueError, TypeError) as exc:
        logger.warning(
            "[translate-scripture] structured parse failed with %s; falling back.",
            exc.__class__.__name__,
        )
        return None

    usage = getattr(resp, "usage", None)
    track_openai_usage(usage, model_name, dependencies.extract_cached_tokens_fn, add_tokens)
    return cast(TranslatedPassage | None, resp.output_parsed)


def _build_verbatim_response(context: TranslationContext) -> dict[str, Any]:
    return {
        "suppress_translation": True,
        "content_language": str(context.resolved_lang),
        "header_is_translated": False,
        "segments": [
            {"type": "header_book", "text": context.canonical_book},
            {"type": "header_suffix", "text": context.header_suffix},
            {"type": "scripture", "text": context.body_source},
        ],
    }


def _build_structured_response(
    context: TranslationContext,
    translated: TranslatedPassage,
) -> dict[str, Any]:
    return {
        "suppress_translation": True,
        "content_language": translated.content_language,
        "header_is_translated": True,
        "segments": [
            {
                "type": "header_book",
                "text": translated.header_book or context.canonical_book,
            },
            {
                "type": "header_suffix",
                "text": translated.header_suffix or context.header_suffix,
            },
            {
                "type": "scripture",
                "text": _normalize_whitespace(translated.body),
            },
        ],
    }


def _build_fallback_response(
    context: TranslationContext,
    dependencies: TranslationDependencies,
    agentic_strength: str,
) -> dict[str, Any]:
    translated_body = _normalize_whitespace(
        dependencies.translate_text_fn(
            response_text=context.body_source,
            target_language=context.target_code,
            agentic_strength=agentic_strength,
        )
    )
    translated_book = dependencies.translate_text_fn(
        response_text=context.canonical_book,
        target_language=context.target_code,
        agentic_strength=agentic_strength,
    )
    return {
        "suppress_translation": True,
        "content_language": context.target_code,
        "header_is_translated": True,
        "segments": [
            {"type": "header_book", "text": translated_book},
            {"type": "header_suffix", "text": context.header_suffix},
            {"type": "scripture", "text": translated_body},
        ],
    }


def _wrap_translation_response(
    context: TranslationContext,
    response_obj: dict[str, Any],
) -> dict[str, Any]:
    return {
        "responses": [{"intent": IntentType.TRANSLATE_SCRIPTURE, "response": response_obj}],
        "passage_followup_context": {
            "intent": IntentType.TRANSLATE_SCRIPTURE,
            "book": context.canonical_book,
            "ranges": [tuple(r) for r in context.ranges],
            "target_language": context.target_code,
        },
    }


def translate_scripture(
    request: TranslationRequestParams,
    dependencies: TranslationDependencies,
) -> dict[str, Any]:
    """Handle translate-scripture: return verses translated into a target language."""
    logger.info(
        "[translate-scripture] start; query_lang=%s; query=%s",
        request.query_lang,
        request.query,
    )

    try:
        context = _build_translation_context(request, dependencies)
    except TranslationContextError as exc:
        return exc.response

    if context.target_code == context.resolved_lang:
        return _wrap_translation_response(context, _build_verbatim_response(context))

    structured = _attempt_structured_translation(request, context, dependencies)

    if structured is not None:
        response_obj = _build_structured_response(context, structured)
    else:
        response_obj = _build_fallback_response(context, dependencies, request.agentic_strength)
    return _wrap_translation_response(context, response_obj)


def get_translation_helps(
    request: TranslationHelpsRequestParams,
    dependencies: TranslationHelpsDependencies,
) -> dict[str, Any]:
    """Generate focused translation helps guidance for a selected passage."""
    logger.info(
        "[translation-helps] start; query_lang=%s; query=%s", request.query_lang, request.query
    )

    payload, error_response = _prepare_translation_helps_payload(request, dependencies)
    if error_response:
        return error_response
    if payload is None:
        return _translation_helps_response(
            "I couldn't prepare translation helps for that selection. Please try again."
        )

    model_name = dependencies.select_model_fn(
        request.agentic_strength, allow_low=True, allow_very_low=True
    )
    raw_helps_digest = hashlib.sha256(
        json.dumps(payload.raw_helps, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    cache_key = TranslationHelpsCacheKey(
        schema=TRANSLATION_HELPS_CACHE_SCHEMA,
        canonical_book=payload.canonical_book,
        ranges=tuple(payload.ranges),
        agentic_strength=request.agentic_strength,
        model_name=model_name,
        raw_helps_digest=raw_helps_digest,
    )

    def _compute_translation_helps() -> dict[str, Any]:
        messages = dependencies.build_messages_fn(
            payload.ref_label,
            payload.context_obj,
        )
        logger.info("[translation-helps] invoking LLM with %d helps", len(payload.raw_helps))
        resp = request.client.responses.create(
            model=model_name,
            instructions=TRANSLATION_HELPS_AGENT_SYSTEM_PROMPT,
            input=cast(Any, messages),
            store=False,
        )
        usage = getattr(resp, "usage", None)
        track_openai_usage(usage, model_name, dependencies.extract_cached_tokens_fn, add_tokens)

        header = f"Translation helps for {payload.ref_label}\n\n"
        response_text = header + (resp.output_text or "")
        logger.info("[translation-helps] done (generated)")
        return {
            "responses": [
                {
                    "intent": IntentType.GET_TRANSLATION_HELPS,
                    "response": response_text,
                }
            ],
            "passage_followup_context": {
                "intent": IntentType.GET_TRANSLATION_HELPS,
                "book": payload.canonical_book,
                "ranges": payload.ranges,
            },
        }

    helps_response, hit = _TRANSLATION_HELPS_CACHE.get_or_set(cache_key, _compute_translation_helps)
    if hit:
        logger.info(
            "[translation-helps] served from cache for book=%s ranges=%s",
            payload.canonical_book,
            payload.ranges,
        )
    if payload.truncation is not None:
        responses = helps_response.get("responses")
        if isinstance(responses, list) and responses:
            notice_payload: dict[str, Any] = {
                "verse_limit": config.TRANSLATION_HELPS_VERSE_LIMIT,
                "delivered_label": payload.ref_label,
            }
            if payload.truncation.original_label:
                notice_payload["original_label"] = payload.truncation.original_label
            responses[0]["truncation_notice"] = notice_payload
    return helps_response


__all__ = [
    "TRANSLATE_PASSAGE_AGENT_SYSTEM_PROMPT",
    "TRANSLATION_HELPS_AGENT_SYSTEM_PROMPT",
    "translate_scripture",
    "get_translation_helps",
]
