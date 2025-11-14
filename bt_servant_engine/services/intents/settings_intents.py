"""Intent handlers for user preference settings (language, agentic strength)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, cast

from openai import OpenAI, OpenAIError
from openai.types.responses.easy_input_message_param import EasyInputMessageParam

from bt_servant_engine.core.agentic import AgenticStrengthChoice, AgenticStrengthSetting
from bt_servant_engine.core.intents import IntentType
from bt_servant_engine.core.language import (
    LANGUAGE_OTHER,
    ResponseLanguage,
    friendly_language_name,
    normalize_language_code,
)
from bt_servant_engine.core.logging import get_logger
from bt_servant_engine.services.openai_utils import extract_cached_input_tokens, track_openai_usage
from utils.identifiers import get_log_safe_user_id
from utils.perf import add_tokens

logger = get_logger(__name__)

SET_RESPONSE_LANGUAGE_AGENT_SYSTEM_PROMPT = """
Task: Determine the language the user wants responses in, based on conversation context and the latest message.

Allowed outputs: any ISO 639-1 language code (e.g., en, fr, tr). If unclear, return Other.

Instructions:
- Use conversation history and the most recent message to infer the user's desired response language.
- Prefer two-letter ISO codes; accept common lowercase variants like "pt-br" when the user is explicit.
- If the request is ambiguous or mentions a language we cannot map to an ISO code, return Other.
- Output must match the provided schema with no additional prose.
"""

SET_AGENTIC_STRENGTH_AGENT_SYSTEM_PROMPT = """
Task: Determine whether the user is asking to adjust the agentic strength setting. The allowed values are "normal",
"low", and "very_low". Use the conversation context and latest message to infer the requested level.

Output schema: { "strength": <normal|low|very_low|unknown> }

Rules:
- If the user clearly requests "normal", "low", or "very low", return that value.
- If the intent is ambiguous or references any other option, return "unknown".
- Do not include explanations or additional text. Only produce a JSON object that matches the schema.
"""


@dataclass(slots=True)
class ResponseLanguageRequest:
    """Inputs required to resolve the user's preferred response language."""

    client: OpenAI
    user_id: str
    user_query: str
    chat_history: list[dict[str, str]]


@dataclass(slots=True)
class ResponseLanguageDependencies:
    """External helpers needed to persist the detected response language."""

    set_user_response_language: Callable[[str, str], Any]


def _conversation_messages(
    chat_history: list[dict[str, str]],
    user_query: str,
    final_prompt: str,
) -> list[EasyInputMessageParam]:
    """Return a shared set of conversation messages for classification prompts."""
    history_json = json.dumps(chat_history)
    return [
        {"role": "user", "content": f"Past conversation: {history_json}"},
        {"role": "user", "content": f"The user's most recent message: {user_query}"},
        {"role": "user", "content": final_prompt},
    ]


def set_response_language(
    request: ResponseLanguageRequest,
    dependencies: ResponseLanguageDependencies,
) -> dict[str, Any]:
    """Detect and persist the user's desired response language."""
    chat_input = _conversation_messages(
        request.chat_history,
        request.user_query,
        "What language is the user trying to set their response language to?",
    )
    response = request.client.responses.parse(
        model="gpt-4o",
        instructions=SET_RESPONSE_LANGUAGE_AGENT_SYSTEM_PROMPT,
        input=cast(Any, chat_input),
        text_format=ResponseLanguage,
        temperature=0,
        store=False,
    )
    usage = getattr(response, "usage", None)
    track_openai_usage(usage, "gpt-4o", extract_cached_input_tokens, add_tokens)
    resp_lang = cast(ResponseLanguage, response.output_parsed)
    if resp_lang.language == LANGUAGE_OTHER:
        response_text = (
            "I can set my responses to any language. "
            "Please mention the specific language or provide its ISO 639-1 code "
            "(for example: en, fr, tr)."
        )
        return {
            "responses": [{"intent": IntentType.SET_RESPONSE_LANGUAGE, "response": response_text}]
        }

    normalized_code = normalize_language_code(resp_lang.language)
    if not normalized_code or normalized_code == LANGUAGE_OTHER:
        response_text = (
            "I wasn't able to determine the language you're requesting. "
            "Please provide a clear ISO 639-1 code so I can save it."
        )
        return {
            "responses": [{"intent": IntentType.SET_RESPONSE_LANGUAGE, "response": response_text}]
        }

    dependencies.set_user_response_language(request.user_id, normalized_code)
    language_name = friendly_language_name(normalized_code, fallback=normalized_code)
    response_text = f"Setting response language to: {language_name}"
    return {
        "responses": [{"intent": IntentType.SET_RESPONSE_LANGUAGE, "response": response_text}],
        "user_response_language": normalized_code,
    }


@dataclass(slots=True)
class AgenticStrengthRequest:
    """Inputs needed to detect the desired agentic strength."""

    client: OpenAI
    user_id: str
    user_query: str
    chat_history: list[dict[str, str]]
    log_pseudonym_secret: str


@dataclass(slots=True)
class AgenticStrengthDependencies:
    """Callback collection for persisting agentic strength preferences."""

    set_user_agentic_strength: Callable[[str, str], Any]


def _parse_agentic_strength(
    request: AgenticStrengthRequest,
) -> AgenticStrengthSetting | None:
    chat_input = _conversation_messages(
        request.chat_history,
        request.user_query,
        "Is the user asking to set the agentic strength to normal, low, or very low?",
    )
    try:
        response = request.client.responses.parse(
            model="gpt-4o",
            instructions=SET_AGENTIC_STRENGTH_AGENT_SYSTEM_PROMPT,
            input=cast(Any, chat_input),
            text_format=AgenticStrengthSetting,
            temperature=0,
            store=False,
        )
        usage = getattr(response, "usage", None)
        track_openai_usage(usage, "gpt-4o", extract_cached_input_tokens, add_tokens)
        return cast(AgenticStrengthSetting | None, response.output_parsed)
    except OpenAIError:
        logger.error(
            "[agentic-strength] OpenAI request failed while parsing user preference.", exc_info=True
        )
        return None
    except (TypeError, ValueError) as exc:
        logger.error(
            "[agentic-strength] Unexpected failure while parsing agentic strength: %s",
            exc.__class__.__name__,
            exc_info=True,
        )
        return None


def set_agentic_strength(
    request: AgenticStrengthRequest,
    dependencies: AgenticStrengthDependencies,
) -> dict[str, Any]:
    """Detect and persist the user's preferred agentic strength."""
    parsed = _parse_agentic_strength(request)

    if not parsed or parsed.strength == AgenticStrengthChoice.UNKNOWN:
        msg = (
            "I can set the agentic strength to normal, low, or very low. "
            "Please specify one of those options so I can update it."
        )
        return {"responses": [{"intent": IntentType.SET_AGENTIC_STRENGTH, "response": msg}]}

    desired = parsed.strength.value
    try:
        dependencies.set_user_agentic_strength(request.user_id, desired)
    except ValueError:
        masked_user_id = get_log_safe_user_id(request.user_id, secret=request.log_pseudonym_secret)
        logger.warning(
            "[agentic-strength] Attempted to set invalid value '%s' for user %s",
            desired,
            masked_user_id,
        )
        msg = (
            "That setting isn't supported. "
            "I can only use normal, low, or very low for agentic strength."
        )
        return {"responses": [{"intent": IntentType.SET_AGENTIC_STRENGTH, "response": msg}]}

    friendly = {
        "normal": "Normal",
        "low": "Low",
        "very_low": "Very Low",
    }.get(desired, desired.capitalize())
    response_text = (
        f"Agentic strength set to {friendly.lower()}. I'll use the {friendly} setting from now on."
    )
    return {
        "responses": [{"intent": IntentType.SET_AGENTIC_STRENGTH, "response": response_text}],
        "agentic_strength": desired,
        "user_agentic_strength": desired,
    }


__all__ = [
    "SET_RESPONSE_LANGUAGE_AGENT_SYSTEM_PROMPT",
    "SET_AGENTIC_STRENGTH_AGENT_SYSTEM_PROMPT",
    "ResponseLanguageRequest",
    "ResponseLanguageDependencies",
    "AgenticStrengthRequest",
    "AgenticStrengthDependencies",
    "set_response_language",
    "set_agentic_strength",
]
