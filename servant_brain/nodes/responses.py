"""Response synthesis, translation, and chunking nodes."""
# pylint: disable=duplicate-code
from __future__ import annotations

import json
from typing import Any, List, cast

from langgraph.graph import END
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from openai.types.responses.easy_input_message_param import EasyInputMessageParam

from config import config
from logger import get_logger
from servant_brain.dependencies import open_ai_client, supported_language_map
from servant_brain.intents import chunk_message as _intent_chunk_message
from servant_brain.intents import translate_responses as _intent_translate_responses
from servant_brain.nodes.language import detect_language
from servant_brain.prompts import (
    CHOP_AGENT_SYSTEM_PROMPT,
    COMBINE_RESPONSES_SYSTEM_PROMPT,
    RESPONSE_TRANSLATOR_SYSTEM_PROMPT,
)
from servant_brain.state import BrainState
from servant_brain.tokens import extract_cached_input_tokens
from utils import chop_text, combine_chunks
from utils.bible_locale import get_book_name
from utils.perf import add_tokens

logger = get_logger(__name__)


def combine_responses(chat_history, latest_user_message, responses) -> str:
    """Ask OpenAI to synthesize multiple node responses into one coherent text."""

    uncombined_responses = json.dumps(responses)
    logger.info("preparing to combine responses:\n\n%s", uncombined_responses)
    messages: list[EasyInputMessageParam] = [
        {"role": "developer", "content": f"conversation history: {chat_history}"},
        {"role": "developer", "content": f"latest user message: {latest_user_message}"},
        {"role": "developer", "content": f"responses to synthesize: {uncombined_responses}"},
    ]
    response = open_ai_client.responses.create(
        model="gpt-4o",
        instructions=COMBINE_RESPONSES_SYSTEM_PROMPT,
        input=cast(Any, messages),
    )
    usage = getattr(response, "usage", None)
    if usage is not None:
        it = getattr(usage, "input_tokens", None)
        ot = getattr(usage, "output_tokens", None)
        tt = getattr(usage, "total_tokens", None)
        if tt is None and (it is not None or ot is not None):
            tt = (it or 0) + (ot or 0)
        cit = extract_cached_input_tokens(usage)
        add_tokens(it, ot, tt, model="gpt-4o", cached_input_tokens=cit)
    combined = response.output_text
    logger.info("combined response from openai: %s", combined)
    return combined


def translate_text(response_text: str, target_language: str) -> str:
    """Translate a single text into the target ISO 639-1 language code."""

    chat_messages = cast(List[ChatCompletionMessageParam], [
        {"role": "system", "content": RESPONSE_TRANSLATOR_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"text to translate: {response_text}\n\n"
                f"ISO 639-1 code representing target language: {target_language}"
            ),
        },
    ])
    completion = open_ai_client.chat.completions.create(
        model="gpt-4o",
        messages=chat_messages,
    )
    usage = getattr(completion, "usage", None)
    if usage is not None:
        it = getattr(usage, "prompt_tokens", None)
        ot = getattr(usage, "completion_tokens", None)
        tt = getattr(usage, "total_tokens", None)
        cit = extract_cached_input_tokens(usage)
        add_tokens(it, ot, tt, model="gpt-4o", cached_input_tokens=cit)
    content = completion.choices[0].message.content
    if isinstance(content, list):
        text = "".join(part.get("text", "") if isinstance(part, dict) else "" for part in content)
    elif content is None:
        text = ""
    else:
        text = content
    logger.info('chunk: \n%s\n\ntranslated to:\n%s', response_text, text)
    return cast(str, text)


def translate_responses(state: Any) -> dict:
    """Delegate to the translate_responses intent helper."""

    return _intent_translate_responses.translate_responses(
        state,
        combine_responses=combine_responses,
        is_protected_response_item=_is_protected_response_item,
        reconstruct_structured_text=_reconstruct_structured_text,
        detect_language=detect_language,
        translate_text=translate_text,
        supported_language_map=supported_language_map,
    )


def chunk_message(state: Any) -> dict:
    """Chunk a message if it exceeds downstream delivery limits."""

    return _intent_chunk_message.chunk_message(
        state,
        chop_text=chop_text,
        combine_chunks=combine_chunks,
        CHOP_AGENT_SYSTEM_PROMPT=CHOP_AGENT_SYSTEM_PROMPT,
        open_ai_client=open_ai_client,
        add_tokens=add_tokens,
        extract_cached_input_tokens=extract_cached_input_tokens,
    )


def needs_chunking(state: BrainState) -> str:
    """Return next node key if chunking is required, otherwise finish."""

    first_response = state["translated_responses"][0]
    if len(first_response) > config.MAX_META_TEXT_LENGTH:
        logger.warning('message to big: %d chars. preparing to chunk.', len(first_response))
        return "chunk_message_node"
    return END


def _is_protected_response_item(item: dict) -> bool:
    body = cast(dict | str, item.get("response"))
    if isinstance(body, dict):
        if body.get("suppress_translation"):
            return True
        if isinstance(body.get("segments"), list):
            segs = cast(list, body.get("segments"))
            return any(isinstance(seg, dict) and seg.get("type") == "scripture" for seg in segs)
    return False


def _reconstruct_structured_text(resp_item: dict | str, localize_to: str | None) -> str:
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
            stype = seg.get("type")
            txt = cast(str, seg.get("text", ""))
            if stype == "header_book":
                header_book = txt
            elif stype == "header_suffix":
                header_suffix = txt
            elif stype == "scripture":
                scripture_text = txt
        book = get_book_name(localize_to or "en", header_book) if localize_to else header_book
        header = (f"{book} {header_suffix}" if header_suffix else book).strip() + ":"
        return header + ("\n\n" + scripture_text if scripture_text else "")
    return str(body)


__all__ = [
    "combine_responses",
    "translate_text",
    "translate_responses",
    "chunk_message",
    "needs_chunking",
]
