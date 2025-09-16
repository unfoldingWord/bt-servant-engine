"""Language detection and response-language nodes."""
# pylint: disable=duplicate-code
from __future__ import annotations

import re
from typing import Any, cast

from openai.types.responses.easy_input_message_param import EasyInputMessageParam

from logger import get_logger
from servant_brain.classifier import IntentType
from servant_brain.dependencies import LANGUAGE_UNKNOWN, open_ai_client, supported_language_map
from servant_brain.language import (
    DETECT_LANGUAGE_AGENT_SYSTEM_PROMPT,
    MessageLanguage,
    ResponseLanguage,
    SET_RESPONSE_LANGUAGE_AGENT_SYSTEM_PROMPT,
)
from servant_brain.state import BrainState
from servant_brain.tokens import extract_cached_input_tokens
from utils.perf import add_tokens
from db import set_user_response_language

logger = get_logger(__name__)

_INSTRUCTION_PATTERN = re.compile(
    r"\b(summarize|explain|what|who|why|how|list|give|provide)\b", re.IGNORECASE
)
_VERSE_PATTERN = re.compile(r"\b[A-Za-z]{2,4}\s+\d+:\d+\b")


def detect_language(text: str) -> str:
    """Detect ISO 639-1 language code of the given text via OpenAI."""

    messages: list[EasyInputMessageParam] = [
        {"role": "user", "content": f"text: {text}"},
    ]
    response = open_ai_client.responses.parse(
        model="gpt-4o",
        instructions=DETECT_LANGUAGE_AGENT_SYSTEM_PROMPT,
        input=cast(Any, messages),
        text_format=MessageLanguage,
        temperature=0,
        store=False,
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
    message_language = cast(MessageLanguage | None, response.output_parsed)
    predicted = message_language.language.value if message_language else "en"
    logger.info("language detection (model): %s", predicted)

    try:
        has_english_instruction = bool(_INSTRUCTION_PATTERN.search(str(text)))
        has_verse_pattern = bool(_VERSE_PATTERN.search(str(text)))
        logger.info(
            "heuristic_guard: predicted=%s english_instruction=%s verse_pattern=%s",
            predicted,
            has_english_instruction,
            has_verse_pattern,
        )
        if predicted == "id" and has_english_instruction and has_verse_pattern:
            logger.info(
                "heuristic_guard: overriding id -> en due to English instruction "
                "+ verse pattern"
            )
            predicted = "en"
    except re.error as err:  # pragma: no cover - defensive guard
        logger.info(
            "heuristic_guard: regex error (%s); keeping model prediction: %s",
            err,
            predicted,
        )

    return predicted


def determine_query_language(state: Any) -> dict:
    """Determine the language of the user's original query and set collection order."""

    s = cast(BrainState, state)
    query = s["user_query"]
    query_language = detect_language(query)
    logger.info("language code %s detected by gpt-4o.", query_language)
    stack_rank_collections = ["knowledgebase", "en_resources"]
    if query_language and query_language not in ("en", LANGUAGE_UNKNOWN):
        localized_collection = f"{query_language}_resources"
        stack_rank_collections.append(localized_collection)
        logger.info(
            "appended localized resources collection: %s (language=%s)",
            localized_collection,
            query_language,
        )

    return {"query_language": query_language, "stack_rank_collections": stack_rank_collections}


def set_response_language(state: Any) -> dict:
    """Update the user's preferred response language when requested."""

    s = cast(BrainState, state)
    chat_history = s["user_chat_history"]
    query = s["user_query"]
    chat_input: list[EasyInputMessageParam] = [
        {
            "role": "user",
            "content": f"conversation history: {chat_history}",
        },
        {
            "role": "user",
            "content": f"current message: {query}",
        },
    ]
    response = open_ai_client.responses.parse(
        model="gpt-4o",
        instructions=SET_RESPONSE_LANGUAGE_AGENT_SYSTEM_PROMPT,
        input=cast(Any, chat_input),
        text_format=ResponseLanguage,
        temperature=0,
        store=False,
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
    resp_lang = cast(ResponseLanguage, response.output_parsed)
    if resp_lang.language.value == "Other":
        supported_language_list = ", ".join(supported_language_map.keys())
        response_text = (
            "I think you're trying to set the response language. The supported languages are: "
            f"{supported_language_list}. "
            "If this is your intent, please clearly tell me which supported language "
            "to use when responding."
        )
        return {
            "responses": [
                {"intent": IntentType.SET_RESPONSE_LANGUAGE, "response": response_text},
            ]
        }
    user_id = cast(str, s["user_id"])
    response_language_code = str(resp_lang.language.value)
    set_user_response_language(user_id, response_language_code)
    language_name = supported_language_map.get(response_language_code, response_language_code)
    response_text = f"Setting response language to: {language_name}"
    return {
        "responses": [
            {"intent": IntentType.SET_RESPONSE_LANGUAGE, "response": response_text},
        ],
        "user_response_language": response_language_code,
    }


__all__ = ["detect_language", "determine_query_language", "set_response_language"]
