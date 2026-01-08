"""Response processing pipeline: translation, combination, and chunking."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Optional, cast

from langgraph.graph import END
from openai import OpenAI, OpenAIError
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam

from bt_servant_engine.core.agentic import ALLOWED_AGENTIC_STRENGTH
from bt_servant_engine.core.config import config
from bt_servant_engine.core.language import LANGUAGE_UNKNOWN
from bt_servant_engine.core.logging import get_logger
from bt_servant_engine.services.openai_utils import track_openai_usage
from bt_servant_engine.services.preprocessing import detect_language as detect_language_impl
from bt_servant_engine.services.response_helpers import (
    normalize_single_response as normalize_single_response_impl,
    sample_for_language_detection as sample_for_language_detection_impl,
)
from utils import chop_text, combine_chunks
from utils.bible_locale import get_book_name
from utils.perf import add_tokens

logger = get_logger(__name__)

COMMA_SPLIT_THRESHOLD = 10

RESPONSE_TRANSLATOR_SYSTEM_PROMPT = (
    "You are a translator for the final output in a chatbot system. "
    "You will receive text that needs to be translated into the language represented by "
    "the specified ISO 639-1 code. ALWAYS translate every part of the input text. NEVER drop, "
    "summarize, or omit sentences, notices, or other content; return the entire text translated "
    "faithfully."
)

CHOP_AGENT_SYSTEM_PROMPT = (
    "You are an agent tasked to ensure that a message intended for Whatsapp fits within the "
    "1500 character limit. Chop the supplied text in the biggest possible semantic chunks, "
    "while making sure no chunk is >= 1500 characters. Your output should be a valid JSON "
    "array containing strings (wrapped in double quotes!!) constituting the chunks. Only "
    "return the json array!! No ```json wrapper or the like. Again, make chunks as big as "
    "possible!!!"
)


@dataclass(slots=True)
class ResponseTranslationDependencies:
    """Callable hooks used to resolve model selection and cached token usage."""

    model_for_agentic_strength: Callable[..., Any]
    extract_cached_input_tokens: Callable[..., Any]


@dataclass(slots=True)
class ResponseTranslationRequest:
    """Inputs required to translate free-form text into a target language."""

    client: OpenAI
    text: str
    target_language: str
    agentic_strength: Optional[str] = None


@dataclass(slots=True)
class ResponseLocalizationRequest:
    """Container describing a localization request for structured responses."""

    client: OpenAI
    response: dict | str
    target_language: str
    agentic_strength: str


@dataclass(slots=True)
class ChunkingDependencies:
    """Shared utilities relied on by chunking helpers."""

    extract_cached_input_tokens: Callable[..., Any]


@dataclass(slots=True)
class ChunkingRequest:
    """Inputs required to chunk a translated response down to message limits."""

    client: OpenAI
    text_to_chunk: str
    additional_responses: list[str]
    chunk_max: int


def reconstruct_structured_text(resp_item: dict | str, localize_to: Optional[str]) -> str:
    """Render a response item to plain text, optionally localizing the header book name.

    - If `resp_item` is a plain string, return it.
    - If structured with segments, rebuild: "<Book> <suffix>:\n\n<scripture>".
    - If `localize_to` is provided, map the book to that language via get_book_name.
      Otherwise fall back to the canonical name.
    """
    if isinstance(resp_item, str):
        return resp_item
    body = cast(dict | str, resp_item.get("response"))
    if isinstance(body, dict) and isinstance(body.get("segments"), list):
        segs = cast(list, body.get("segments"))
        header_book = ""
        header_suffix = ""
        scripture_text = ""
        for seg in segs:
            if not isinstance(seg, dict):
                continue
            st = seg.get("type")
            txt = cast(str, seg.get("text", ""))
            if st == "header_book":
                header_book = txt
            elif st == "header_suffix":
                header_suffix = txt
            elif st == "scripture":
                scripture_text = txt
        book = get_book_name(localize_to or "en", header_book) if localize_to else header_book
        header = (f"{book} {header_suffix}" if header_suffix else book).strip() + ":"
        return header + ("\n\n" + scripture_text if scripture_text else "")
    return str(body)


def translate_text(
    request: ResponseTranslationRequest,
    dependencies: ResponseTranslationDependencies,
) -> str:
    """Translate a single text into the target ISO 639-1 language code.

    Returns a plain string. If the OpenAI SDK returns a structured content
    list or None, normalize it to a string.
    """
    resolved_strength = (
        request.agentic_strength if request.agentic_strength in ALLOWED_AGENTIC_STRENGTH else None
    )
    if resolved_strength is None:
        configured = getattr(config, "AGENTIC_STRENGTH", "normal")
        resolved_strength = configured if configured in ALLOWED_AGENTIC_STRENGTH else "normal"
    model_name = dependencies.model_for_agentic_strength(
        resolved_strength, allow_low=False, allow_very_low=True
    )
    chat_messages = cast(
        list[ChatCompletionMessageParam],
        [
            {
                "role": "system",
                "content": RESPONSE_TRANSLATOR_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": (
                    f"text to translate: {request.text}\n\n"
                    f"ISO 639-1 code representing target language: {request.target_language}"
                ),
            },
        ],
    )
    completion = request.client.chat.completions.create(
        model=model_name,
        messages=chat_messages,
    )
    usage = getattr(completion, "usage", None)
    track_openai_usage(usage, model_name, dependencies.extract_cached_input_tokens, add_tokens)
    content = completion.choices[0].message.content
    if isinstance(content, list):
        text = "".join(part.get("text", "") if isinstance(part, dict) else "" for part in content)
    elif content is None:
        text = ""
    else:
        text = content
    logger.info("chunk: \n%s\n\ntranslated to:\n%s", request.text, text)
    return cast(str, text)


def translate_or_localize_response(
    request: ResponseLocalizationRequest,
    dependencies: ResponseTranslationDependencies,
) -> str:
    """Translate free-form text or localize structured scripture outputs."""
    placeholder_key = getattr(request.client, "api_key", None) == "sk-test"
    detection_fn = detect_language_impl
    detection_is_patched = detection_fn.__module__ != "bt_servant_engine.services.preprocessing"
    if isinstance(request.response, str):
        sample = sample_for_language_detection_impl(request.response)
        detected_lang = (
            detect_language_impl(request.client, sample, agentic_strength=request.agentic_strength)
            if sample and (detection_is_patched or not placeholder_key)
            else request.target_language
        )
        if detected_lang != request.target_language:
            logger.info("preparing to translate to %s", request.target_language)
            return translate_text(
                ResponseTranslationRequest(
                    client=request.client,
                    text=request.response,
                    target_language=request.target_language,
                    agentic_strength=request.agentic_strength,
                ),
                dependencies,
            )
        logger.info("chunk translation not required. using chunk as is.")
        return request.response

    body = cast(dict | str, request.response.get("response"))
    if isinstance(body, dict) and isinstance(body.get("segments"), list):
        item_lang = cast(Optional[str], body.get("content_language"))
        header_is_translated = bool(body.get("header_is_translated"))
        localize_to = None if header_is_translated else (item_lang or request.target_language)
        return reconstruct_structured_text(resp_item=request.response, localize_to=localize_to)
    return str(body)


def build_translation_queue(
    state: dict[str, Any],
    protected_items: list[dict],
    normal_items: list[dict],
) -> list[dict | str]:
    """Assemble responses in the order they should be translated or localized.

    With sequential intent processing, we should only ever have one response item.
    """
    _ = state  # Unused but kept for signature compatibility
    queue: list[dict | str] = list(protected_items)

    # Process all normal items in order
    for item in normal_items:
        queue.append(normalize_single_response_impl(item))

    return queue


def resolve_target_language(
    state: dict[str, Any],
    responses_for_translation: list[dict | str],
) -> tuple[Optional[str], Optional[list[str]]]:
    """Determine the target language or build a pass-through fallback."""
    user_language = cast(Optional[str], state.get("user_response_language"))
    if user_language:
        return user_language, None

    target_language = cast(str, state.get("query_language"))
    if target_language != LANGUAGE_UNKNOWN:
        return target_language, None

    logger.warning("target language unknown. bailing out.")
    passthrough_texts: list[str] = [
        reconstruct_structured_text(resp_item=resp, localize_to=None)
        for resp in responses_for_translation
    ]
    notice = (
        "You haven't set your desired response language and I couldn't determine the "
        "language of your original message. Tell me something like "
        "'Set my response language to Turkish (tr)' and I'll use that for future replies."
    )
    passthrough_texts.append(notice)
    return None, passthrough_texts


def chunk_message(
    request: ChunkingRequest,
    dependencies: ChunkingDependencies,
) -> list[str]:
    """Chunk oversized responses to respect WhatsApp limits, via LLM or fallback."""
    logger.info("MESSAGE TOO BIG. CHUNKING...")
    try:
        chat_messages = cast(
            list[ChatCompletionMessageParam],
            [
                {
                    "role": "system",
                    "content": CHOP_AGENT_SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": f"text to chop: \n\n{request.text_to_chunk}",
                },
            ],
        )
        completion = request.client.chat.completions.create(
            model="gpt-4o",
            messages=chat_messages,
        )
        usage = getattr(completion, "usage", None)
        track_openai_usage(usage, "gpt-4o", dependencies.extract_cached_input_tokens, add_tokens)
        response_content = completion.choices[0].message.content
        if not isinstance(response_content, str):
            raise ValueError("empty or non-text content from chat completion")
        chunks = json.loads(response_content)
    except (OpenAIError, json.JSONDecodeError, ValueError):
        logger.error("LLM chunking failed. Falling back to deterministic chunking.", exc_info=True)
        chunks = None

    # Deterministic safeguards: if LLM returned a single massive chunk or invalid shape,
    # or if we skipped to fallback
    def _pack_items(items: list[str], max_len: int) -> list[str]:
        out: list[str] = []
        cur = ""
        for it in items:
            sep = ", " if cur else ""
            if len(cur) + len(sep) + len(it) <= max_len:
                cur += sep + it
            else:
                if cur:
                    out.append(cur)
                if len(it) <= max_len:
                    cur = it
                else:
                    # hard-split this long token
                    for j in range(0, len(it), max_len):
                        out.append(it[j : j + max_len])
                    cur = ""
        if cur:
            out.append(cur)
        return out

    if not isinstance(chunks, list) or any(not isinstance(c, str) for c in chunks):
        # Try delimiter-aware fallback for comma-heavy lists first
        if request.text_to_chunk.count(",") >= COMMA_SPLIT_THRESHOLD:
            parts = [p.strip() for p in request.text_to_chunk.split(",") if p.strip()]
            chunks = _pack_items(parts, request.chunk_max)
        else:
            chunks = chop_text(text=request.text_to_chunk, n=request.chunk_max)
    else:
        # Ensure each chunk respects the limit; if not, re-split deterministically
        fixed: list[str] = []
        for c in chunks:
            if len(c) <= request.chunk_max:
                fixed.append(c)
            elif c.count(",") >= COMMA_SPLIT_THRESHOLD:
                parts = [p.strip() for p in c.split(",") if p.strip()]
                fixed.extend(_pack_items(parts, request.chunk_max))
            else:
                fixed.extend(chop_text(text=c, n=request.chunk_max))
        chunks = fixed

    chunks.extend(request.additional_responses)
    return combine_chunks(chunks=chunks, chunk_max=request.chunk_max)


def needs_chunking(state: dict[str, Any]) -> str:
    """Return next node key if chunking is required, otherwise finish."""
    responses = state.get("translated_responses", [])
    if not responses:
        logger.info("[chunk-check] no text responses to send; skipping chunking")
        return END
    first_response = responses[0]
    if len(first_response) > config.MAX_RESPONSE_CHUNK_SIZE:
        logger.warning("message to big: %d chars. preparing to chunk.", len(first_response))
        return "chunk_message_node"
    return END


__all__ = [
    "RESPONSE_TRANSLATOR_SYSTEM_PROMPT",
    "CHOP_AGENT_SYSTEM_PROMPT",
    "ResponseTranslationDependencies",
    "ResponseTranslationRequest",
    "ResponseLocalizationRequest",
    "ChunkingDependencies",
    "ChunkingRequest",
    "reconstruct_structured_text",
    "translate_text",
    "translate_or_localize_response",
    "build_translation_queue",
    "resolve_target_language",
    "chunk_message",
    "needs_chunking",
]
