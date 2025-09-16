"""Conversation-adjacent nodes (greetings, help, unsupported functions)."""
# pylint: disable=duplicate-code
from __future__ import annotations

import json
from typing import Any, cast

from openai.types.responses.easy_input_message_param import EasyInputMessageParam

from logger import get_logger
from servant_brain.classifier import IntentType
from servant_brain.dependencies import open_ai_client
from servant_brain.nodes.capabilities import (
    CONVERSE_AGENT_SYSTEM_PROMPT,
    FIRST_INTERACTION_MESSAGE,
    HELP_AGENT_SYSTEM_PROMPT,
    UNSUPPORTED_FUNCTION_AGENT_SYSTEM_PROMPT,
)
from servant_brain.state import BrainState
from servant_brain.tokens import extract_cached_input_tokens
from db import is_first_interaction, set_first_interaction
from utils.perf import add_tokens

logger = get_logger(__name__)


def start(state: Any) -> dict:
    """Handle first interaction greeting, otherwise no-op."""

    s = cast(BrainState, state)
    user_id = s["user_id"]
    if is_first_interaction(user_id):
        set_first_interaction(user_id, False)
        return {
            "responses": [
                {"intent": "first-interaction", "response": FIRST_INTERACTION_MESSAGE},
            ]
        }
    return {}


def handle_unsupported_function(state: Any) -> dict:
    """Respond when the user asks for an unsupported capability."""

    s = cast(BrainState, state)
    query = s["user_query"]
    chat_history = s["user_chat_history"]
    messages: list[EasyInputMessageParam] = [
        {
            "role": "developer",
            "content": f"Conversation history to use if needed: {json.dumps(chat_history)}",
        },
        {
            "role": "user",
            "content": query,
        },
    ]
    response = open_ai_client.responses.create(
        model="gpt-4o",
        instructions=UNSUPPORTED_FUNCTION_AGENT_SYSTEM_PROMPT,
        input=cast(Any, messages),
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
    resp_text = response.output_text
    logger.info("converse_with_bt_servant response from openai: %s", resp_text)
    return {
        "responses": [
            {"intent": IntentType.PERFORM_UNSUPPORTED_FUNCTION, "response": resp_text},
        ]
    }


def handle_system_information_request(state: Any) -> dict:
    """Provide help/about information for the BT Servant system."""

    s = cast(BrainState, state)
    query = s["user_query"]
    chat_history = s["user_chat_history"]
    messages: list[EasyInputMessageParam] = [
        {
            "role": "developer",
            "content": f"Conversation history to use if needed: {json.dumps(chat_history)}",
        },
        {
            "role": "user",
            "content": query,
        },
    ]
    response = open_ai_client.responses.create(
        model="gpt-4o",
        instructions=HELP_AGENT_SYSTEM_PROMPT,
        input=cast(Any, messages),
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
    resp_text = response.output_text
    logger.info("help response from openai: %s", resp_text)
    return {
        "responses": [
            {"intent": IntentType.RETRIEVE_SYSTEM_INFORMATION, "response": resp_text},
        ]
    }


def converse_with_bt_servant(state: Any) -> dict:
    """Respond conversationally to the user based on context and history."""

    s = cast(BrainState, state)
    query = s["user_query"]
    chat_history = s["user_chat_history"]
    messages: list[EasyInputMessageParam] = [
        {
            "role": "developer",
            "content": f"Conversation history to use if needed: {json.dumps(chat_history)}",
        },
        {
            "role": "user",
            "content": query,
        },
    ]
    response = open_ai_client.responses.create(
        model="gpt-4o",
        instructions=CONVERSE_AGENT_SYSTEM_PROMPT,
        input=cast(Any, messages),
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
    resp_text = response.output_text
    logger.info("converse_with_bt_servant response from openai: %s", resp_text)
    return {
        "responses": [
            {"intent": IntentType.CONVERSE_WITH_BT_SERVANT, "response": resp_text},
        ]
    }


__all__ = [
    "start",
    "handle_unsupported_function",
    "handle_system_information_request",
    "converse_with_bt_servant",
]
