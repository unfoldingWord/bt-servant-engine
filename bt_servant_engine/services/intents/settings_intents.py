"""Intent handlers for user preference settings (language, agentic strength)."""

from __future__ import annotations

import json
from typing import Any, cast

from openai import OpenAI, OpenAIError
from openai.types.responses.easy_input_message_param import EasyInputMessageParam

from bt_servant_engine.core.agentic import AgenticStrengthChoice, AgenticStrengthSetting
from bt_servant_engine.core.intents import IntentType
from bt_servant_engine.core.language import Language, ResponseLanguage
from bt_servant_engine.core.logging import get_logger
from bt_servant_engine.services.openai_utils import extract_cached_input_tokens
from utils.identifiers import get_log_safe_user_id
from utils.perf import add_tokens

logger = get_logger(__name__)

SET_RESPONSE_LANGUAGE_AGENT_SYSTEM_PROMPT = """
Task: Determine the language the user wants responses in, based on conversation context and the latest message.

Allowed outputs: en, ar, fr, es, hi, ru, id, sw, pt, zh, nl, Other

Instructions:
- Use conversation history and the most recent message to infer the user's desired response language.
- Only return one of the allowed outputs. If unclear or unsupported, return Other.
- Consider explicit requests like "reply in French" or language names/codes.
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


def set_response_language(
    client: OpenAI,
    user_id: str,
    user_query: str,
    chat_history: list[dict[str, str]],
    supported_language_map: dict[str, str],
    set_user_response_language_fn: callable,
) -> dict[str, Any]:
    """Detect and persist the user's desired response language."""
    chat_input: list[EasyInputMessageParam] = [
        {
            "role": "user",
            "content": f"Past conversation: {json.dumps(chat_history)}",
        },
        {
            "role": "user",
            "content": f"the user's most recent message: {user_query}",
        },
        {
            "role": "user",
            "content": "What language is the user trying to set their response language to?",
        },
    ]
    response = client.responses.parse(
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
    if resp_lang.language == Language.OTHER:
        supported_language_list = ", ".join(supported_language_map.keys())
        response_text = (
            f"I think you're trying to set the response language. The supported languages "
            f"are: {supported_language_list}. If this is your intent, please clearly tell "
            f"me which supported language to use when responding."
        )
        return {
            "responses": [{"intent": IntentType.SET_RESPONSE_LANGUAGE, "response": response_text}]
        }
    response_language_code: str = str(resp_lang.language.value)
    set_user_response_language_fn(user_id, response_language_code)
    language_name: str = supported_language_map.get(response_language_code, response_language_code)
    response_text = f"Setting response language to: {language_name}"
    return {
        "responses": [{"intent": IntentType.SET_RESPONSE_LANGUAGE, "response": response_text}],
        "user_response_language": response_language_code,
    }


def set_agentic_strength(
    client: OpenAI,
    user_id: str,
    user_query: str,
    chat_history: list[dict[str, str]],
    set_user_agentic_strength_fn: callable,
    log_pseudonym_secret: str,
) -> dict[str, Any]:
    """Detect and persist the user's preferred agentic strength."""
    chat_input: list[EasyInputMessageParam] = [
        {
            "role": "user",
            "content": f"Past conversation: {json.dumps(chat_history)}",
        },
        {
            "role": "user",
            "content": f"the user's most recent message: {user_query}",
        },
        {
            "role": "user",
            "content": "Is the user asking to set the agentic strength to normal, low, or very low?",
        },
    ]

    try:
        response = client.responses.parse(
            model="gpt-4o",
            instructions=SET_AGENTIC_STRENGTH_AGENT_SYSTEM_PROMPT,
            input=cast(Any, chat_input),
            text_format=AgenticStrengthSetting,
            temperature=0,
            store=False,
        )
        usage = getattr(response, "usage", None)
        if usage is not None:
            add_tokens(
                getattr(usage, "input_tokens", None),
                getattr(usage, "output_tokens", None),
                getattr(usage, "total_tokens", None)
                or (
                    (getattr(usage, "input_tokens", None) or 0)
                    + (getattr(usage, "output_tokens", None) or 0)
                ),
                model="gpt-4o",
                cached_input_tokens=extract_cached_input_tokens(usage),
            )
        parsed = cast(AgenticStrengthSetting | None, response.output_parsed)
    except OpenAIError:
        logger.error(
            "[agentic-strength] OpenAI request failed while parsing user preference.", exc_info=True
        )
        parsed = None
    except Exception:  # pylint: disable=broad-except
        logger.error(
            "[agentic-strength] Unexpected failure while parsing agentic strength.", exc_info=True
        )
        parsed = None

    if not parsed or parsed.strength == AgenticStrengthChoice.UNKNOWN:
        msg = (
            "I can set the agentic strength to normal, low, or very low. Please specify one of those options "
            "so I can update it."
        )
        return {"responses": [{"intent": IntentType.SET_AGENTIC_STRENGTH, "response": msg}]}

    desired = parsed.strength.value
    try:
        set_user_agentic_strength_fn(user_id, desired)
    except ValueError:
        masked_user_id = get_log_safe_user_id(user_id, secret=log_pseudonym_secret)
        logger.warning(
            "[agentic-strength] Attempted to set invalid value '%s' for user %s",
            desired,
            masked_user_id,
        )
        msg = "That setting isn't supported. I can only use normal, low, or very low for agentic strength."
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
    "set_response_language",
    "set_agentic_strength",
]
