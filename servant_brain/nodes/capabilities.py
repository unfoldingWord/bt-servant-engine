"""Capability metadata and prompt templates used by conversation nodes."""
from __future__ import annotations

from typing import List

from servant_brain.classifier import IntentType
from servant_brain.state import Capability


def capabilities() -> List[Capability]:
    """Return the list of user-facing capabilities with examples."""

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

    caps = [c for c in capabilities() if c.get("include_in_boilerplate")]
    lines: list[str] = ["Here’s what I can do:"]
    for idx, cap in enumerate(caps, start=1):
        example = cap["examples"][0] if cap.get("examples") else ""
        lines.append(f"{idx}) {cap['label']} (e.g., '{example}')")
    lines.append("Which would you like me to do?")
    return "\n".join(lines)


def build_full_help_message() -> str:
    """Build a full help message with descriptions and examples for each capability."""

    lines: list[str] = ["Features:"]
    for idx, cap in enumerate(capabilities(), start=1):
        lines.append(f"{idx}. {cap['label']}: {cap['description']}")
        if cap.get("examples"):
            for example in cap["examples"]:
                lines.append(f"   - Example: '{example}'")
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


__all__ = [
    "capabilities",
    "build_boilerplate_message",
    "build_full_help_message",
    "BOILER_PLATE_AVAILABLE_FEATURES_MESSAGE",
    "FULL_HELP_MESSAGE",
    "FIRST_INTERACTION_MESSAGE",
    "UNSUPPORTED_FUNCTION_AGENT_SYSTEM_PROMPT",
    "CONVERSE_AGENT_SYSTEM_PROMPT",
    "HELP_AGENT_SYSTEM_PROMPT",
]
