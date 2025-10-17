"""Simple intent handlers: conversational, help, and unsupported function requests."""

from __future__ import annotations

import json
from typing import Any, List, TypedDict, cast

from openai import OpenAI
from openai.types.responses.easy_input_message_param import EasyInputMessageParam

from bt_servant_engine import BT_SERVANT_RELEASES_URL, BT_SERVANT_VERSION
from bt_servant_engine.core.intents import IntentType
from bt_servant_engine.core.logging import get_logger
from bt_servant_engine.services.openai_utils import extract_cached_input_tokens, track_openai_usage
from utils.perf import add_tokens

logger = get_logger(__name__)


class Capability(TypedDict, total=False):
    """Type for capability definitions."""

    intent: IntentType
    label: str
    description: str
    examples: list[str]
    include_in_boilerplate: bool
    developer_only: bool


def get_capabilities() -> List[Capability]:
    """Return the list of user-facing capabilities with examples.

    Centralized here so greetings, help, and unsupported-function prompts stay in sync
    as new intents are added. Update this list to change feature messaging.
    """
    return [
        {
            "intent": IntentType.GET_PASSAGE_SUMMARY,
            "label": "Summarize a passage",
            "description": "Summarize books, chapters, or verse ranges.",
            "examples": [
                "Summarize Titus 1.",
                "Summarize Mark 1:1–8.",
            ],
            "include_in_boilerplate": True,
        },
        {
            "intent": IntentType.CONSULT_FIA_RESOURCES,
            "label": "FIA process guidance",
            "description": "Use FIA resources to explain the workflow or apply steps to a passage.",
            "examples": [
                "What are the steps of the FIA process?",
            ],
            "include_in_boilerplate": True,
        },
        {
            "intent": IntentType.GET_TRANSLATION_HELPS,
            "label": "Translation helps",
            "description": "Point out typical translation challenges.",
            "examples": [
                "Translation challenges for John 1:1?",
            ],
            "include_in_boilerplate": True,
        },
        {
            "intent": IntentType.GET_PASSAGE_KEYWORDS,
            "label": "Keywords",
            "description": "List key terms in a passage.",
            "examples": [
                "Important words in Romans 1.",
            ],
            "include_in_boilerplate": True,
        },
        {
            "intent": IntentType.RETRIEVE_SCRIPTURE,
            "label": "Show scripture text",
            "description": "Display the verse text for a selection.",
            "examples": [
                "Show John 3:16–18.",
            ],
            "include_in_boilerplate": True,
        },
        {
            "intent": IntentType.LISTEN_TO_SCRIPTURE,
            "label": "Read aloud",
            "description": "Hear the passage as audio.",
            "examples": [
                "Read Romans 8:1–4 aloud.",
            ],
            "include_in_boilerplate": True,
        },
        {
            "intent": IntentType.TRANSLATE_SCRIPTURE,
            "label": "Translate scripture",
            "description": "Translate a passage into another language.",
            "examples": [
                "Translate John 3:16 into Indonesian.",
            ],
            "include_in_boilerplate": True,
        },
        {
            "intent": IntentType.SET_AGENTIC_STRENGTH,
            "label": "Adjust agentic strength",
            "description": "Tune how assertive the assistant should be (normal, low, or very low).",
            "examples": [
                "Set my agentic strength to low.",
            ],
            "include_in_boilerplate": False,
            "developer_only": True,
        },
        {
            "intent": IntentType.SET_RESPONSE_LANGUAGE,
            "label": "Set response language",
            "description": "Choose your preferred reply language.",
            "examples": [
                "Set my response language to Spanish.",
            ],
            "include_in_boilerplate": True,
        },
    ]


def build_boilerplate_message() -> str:
    """Build a concise 'what I can do' list with examples."""
    caps = [
        c
        for c in get_capabilities()
        if c.get("include_in_boilerplate") and not c.get("developer_only", False)
    ]
    lines: list[str] = ["Here's what I can do:"]
    for c in caps:
        example = c["examples"][0] if c.get("examples") else ""
        lines.append(f"- {c['label']} (e.g., '{example}')")
    lines.append("Which would you like me to do?")
    return "\n".join(lines)


def build_full_help_message() -> str:
    """Build a full help message with descriptions and examples for each capability."""
    lines: list[str] = ["Features:"]
    visible_caps = [c for c in get_capabilities() if not c.get("developer_only", False)]
    for c in visible_caps:
        lines.append(f"- {c['label']}: {c['description']}")
        if c.get("examples"):
            for ex in c["examples"]:
                lines.append(f"   - Example: '{ex}'")
    return "\n".join(lines)


BOILER_PLATE_AVAILABLE_FEATURES_MESSAGE = build_boilerplate_message()
FULL_HELP_MESSAGE = build_full_help_message()

FIRST_INTERACTION_MESSAGE = f"""
Hello! I am the BT Servant. This is our first conversation. Let's work together to understand and translate God's word!

{BOILER_PLATE_AVAILABLE_FEATURES_MESSAGE}
"""

UNSUPPORTED_FUNCTION_AGENT_SYSTEM_PROMPT = f"""
# Identity

You are a part of a RAG bot system that assists Bible translators. You are one node in the decision/intent processing
lang graph. Specifically, your job is to handle the perform-unsupported-function intent. This means the user is trying
to perform an unsupported function.

# Instructions

Respond appropriately to the user's request to do something that you currently can't do. Leverage the
user's message and the conversation history if needed. Make sure to always end your response with some version of
the boiler plate available features message (see below).

<boiler_plate_available_features_message>
    {BOILER_PLATE_AVAILABLE_FEATURES_MESSAGE}
</boiler_plate_available_features_message>
"""

CONVERSE_AGENT_SYSTEM_PROMPT = f"""
# Identity

You are a part of a RAG bot system that assists Bible translators. You are one node in the decision/intent processing
lang graph. Specifically, your job is to handle the converse-with-bt-servant intent by responding conversationally to
the user based on the provided context.

# Instructions

If we are here in the decision graph, the converse-with-bt-servant intent has been detected. You will be provided with
the user's most recent message and conversation history. Your job is to respond conversationally to the user. Unless it
doesn't make sense to do so, aim to end your response with some version of  the boiler plate available features message
(see below).

<boiler_plate_available_features_message>
    {BOILER_PLATE_AVAILABLE_FEATURES_MESSAGE}
</boiler_plate_available_features_message>
"""

HELP_AGENT_SYSTEM_PROMPT = f"""
# Identity

You are a part of a WhatsApp RAG bot system that assists Bible translators called BT Servant. You sole purpose is to
provide help information about the BT Servant system. If this node has been hit, it means the system has already
classified the user's most recent message as a desire to receive help or more information about the system. This is
typically the result of them saying something like: 'help!' or 'tell me about yourself' or 'how does this work?' Thus,
make sure to always provide some help, to the best of your abilities. Always provide help to the user.

# Instructions
You will be supplied with the user's most recent message and also past conversation history. Using this context,
provide the user with information detailing how the system works (the features of the BT Servant system). Use the
feature information below. End your response with a single question inviting the user to pick one capability
(for example: 'Which of these would you like me to do?').

<features_full_help_message>
{FULL_HELP_MESSAGE}
</features_full_help_message>

# Using prior history for better responses

Here are some guidelines for using history for better responses:
1. If you detect in conversation history that you've already said hello, there's no need to say it again.
2. If it doesn't make sense to say "hello!" to the user, based on their most recent message, there's no need to say
'Hello!  I'm here to assist with Bible translation tasks' again.
"""


# ========== Intent Handlers ==========


def handle_unsupported_function(
    client: OpenAI, query: str, chat_history: list[dict[str, str]]
) -> dict[str, Any]:
    """Generate a helpful response when the user requests unsupported functionality."""
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
    response = client.responses.create(
        model="gpt-4o",
        instructions=UNSUPPORTED_FUNCTION_AGENT_SYSTEM_PROMPT,
        input=cast(Any, messages),
        store=False,
    )
    usage = getattr(response, "usage", None)
    track_openai_usage(usage, "gpt-4o", extract_cached_input_tokens, add_tokens)
    unsupported_function_response_text = response.output_text
    logger.info(
        "perform_unsupported_function response from openai: %s",
        unsupported_function_response_text,
    )
    return {
        "responses": [
            {
                "intent": IntentType.PERFORM_UNSUPPORTED_FUNCTION,
                "response": unsupported_function_response_text,
            }
        ]
    }


def handle_system_information_request(
    client: OpenAI, query: str, chat_history: list[dict[str, str]]
) -> dict[str, Any]:
    """Provide help/about information for the BT Servant system."""
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
    response = client.responses.create(
        model="gpt-4o",
        instructions=HELP_AGENT_SYSTEM_PROMPT,
        input=cast(Any, messages),
        store=False,
    )
    usage = getattr(response, "usage", None)
    track_openai_usage(usage, "gpt-4o", extract_cached_input_tokens, add_tokens)
    help_response_text = response.output_text.strip()
    version_tag = f"v{BT_SERVANT_VERSION}"
    if version_tag not in help_response_text:
        version_line = (
            f"🚀 Current version: {version_tag} (release notes: {BT_SERVANT_RELEASES_URL})"
        )
        marker = "The BT Servant system"
        if marker in help_response_text:
            help_response_text = help_response_text.replace(
                marker, f"{version_line}\n\n{marker}", 1
            )
        else:
            help_response_text = f"{version_line}\n\n{help_response_text}"
    logger.info("help response from openai: %s", help_response_text)
    return {
        "responses": [
            {"intent": IntentType.RETRIEVE_SYSTEM_INFORMATION, "response": help_response_text}
        ]
    }


def converse_with_bt_servant(
    client: OpenAI, query: str, chat_history: list[dict[str, str]]
) -> dict[str, Any]:
    """Respond conversationally to the user based on context and history."""
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
    response = client.responses.create(
        model="gpt-4o",
        instructions=CONVERSE_AGENT_SYSTEM_PROMPT,
        input=cast(Any, messages),
        store=False,
    )
    usage = getattr(response, "usage", None)
    track_openai_usage(usage, "gpt-4o", extract_cached_input_tokens, add_tokens)
    converse_response_text = response.output_text
    logger.info("converse_with_bt_servant response from openai: %s", converse_response_text)
    return {
        "responses": [
            {"intent": IntentType.CONVERSE_WITH_BT_SERVANT, "response": converse_response_text}
        ]
    }


__all__ = [
    "Capability",
    "get_capabilities",
    "build_boilerplate_message",
    "build_full_help_message",
    "BOILER_PLATE_AVAILABLE_FEATURES_MESSAGE",
    "FULL_HELP_MESSAGE",
    "FIRST_INTERACTION_MESSAGE",
    "UNSUPPORTED_FUNCTION_AGENT_SYSTEM_PROMPT",
    "CONVERSE_AGENT_SYSTEM_PROMPT",
    "HELP_AGENT_SYSTEM_PROMPT",
    "handle_unsupported_function",
    "handle_system_information_request",
    "converse_with_bt_servant",
]
