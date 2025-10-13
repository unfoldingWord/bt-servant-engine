"""Query preprocessing, language detection, and intent classification."""
# pylint: disable=too-many-lines

from __future__ import annotations

import json
import re
from typing import Any, Optional, cast

from openai import OpenAI
from openai.types.responses.easy_input_message_param import EasyInputMessageParam
from pydantic import BaseModel

from bt_servant_engine.core.agentic import ALLOWED_AGENTIC_STRENGTH
from bt_servant_engine.core.config import config
from bt_servant_engine.core.intents import (
    IntentType,
    IntentWithContext,
    UserIntents,
    UserIntentsStructured,
)
from bt_servant_engine.core.language import (
    Language,
    MessageLanguage,
)
from bt_servant_engine.core.logging import get_logger
from bt_servant_engine.services.openai_utils import (
    extract_cached_input_tokens as _extract_cached_input_tokens,
    track_openai_usage,
)
from utils.perf import add_tokens

logger = get_logger(__name__)

# ========== PROMPTS ==========

PREPROCESSOR_AGENT_SYSTEM_PROMPT = """
# Identity

You are a preprocessor agent/node in a retrieval augmented generation (RAG) pipeline.

# Instructions

Use past conversation context,
if supplied and applicable, to disambiguate or clarify the intent or meaning of the user's current message. Change
as little as possible. Change nothing unless necessary. Only expand or clarify when you are absolutely certain it is
required to understand the user's intent. If there is any uncertainty, leave the message exactly as provided. If the
intent of the user's message is already clear, change nothing. Never greatly expand the user's current message. Changes
should be small or none. Feel free to fix
obvious spelling mistakes or errors, but not logic errors like incorrect books of the Bible. Do NOT narrow the scope of
explicit scripture selections: if a user requests multiple chapters, verse ranges, or disjoint selections (including
conjunctions like "and" or comma/semicolon lists), preserve them exactly as written. If the system has constraints
(for example, only a single chapter can be processed at a time), do NOT modify the user's message to fit those
constraints — leave the message intact and let downstream nodes handle any rejection or guidance. For translation
requests, do NOT add or change a target language; preserve only what the user explicitly stated.

When the immediately preceding assistant response lists numbered capabilities (for example, "1. Summarize a passage"
through "8. Set response language") and the user replies with a message that is just one of those numbers (optionally
with whitespace or punctuation), expand it into the canonical request below, always referencing John 1:1 where
scripture is involved:

- 1 → "Summarize John 1:1."
- 2 → "Guide me through the FIA process for John 1:1."
- 3 → "Provide translation helps for John 1:1."
- 4 → "List the key terms in John 1:1."
- 5 → "Show me John 1:1."
- 6 → "I want to listen to John 1:1."
- 7 → "Translate John 1:1 into Spanish."
- 8 → "Set my response language to Spanish."

Return the clarified
message and the reasons for clarifying or reasons for not changing anything. Examples below.

# Examples

## Example 1

<past_conversation>
    user_message: Summarize the book of Titus.
    assistant_response: The book of titus is about...
</past_conversation>

<current_message>
    user_message: Now Mark
</current_message>

<assistant_response>
    new_message: Now Summarize the book of Mark.
    reason_for_decision: Based on previous context, the user wants the system to do the same thing, but this time
                         with Mark.
    message_changed: True
</assistant_response>

## Example 2

<past_conversation>
    user_message: What is going on in 1 Peter 3:7?
    assistant_response: Peter is instructing Christian husbands to be loving to their wives.
</past_conversation>

<current_message>
    user_message: Summarize Mark 3:1
</current_message>

<assistant_response>
    new_message: Summarize Mark 3:1.
    reason_for_decision: Nothing was changed. The user's current command has nothing to do with past context and
                         is fine as is.
    message_changed: False
</assistant_response>

## Example 3

<past_conversation>
    user_message: Explain John 1:1
    assistant_response: John claims that Jesus, the Word, existed in the beginning with God the Father.
</past_conversation>

<current_message>
    user_message: Explain John 1:3
</current_message>

<assistant_response>
    new_message: Explain John 1:3.
    reason_for_decision: The word 'John' was misspelled in the message.
    message_changed: True
</assistant_response>

## Example 4

<past_conversation>
    assistant_response: Certainly! Here’s what I can do to assist with Bible translation tasks:

        1. Summarize a passage
        2. FIA process guidance
        3. Translation helps
        4. Keywords
        5. Show scripture text
        6. Read aloud
        7. Translate scripture
        8. Set response language

    Which of these capabilities would you like to explore?
</past_conversation>

<current_message>
    user_message: 1
</current_message>

<assistant_response>
    new_message: Summarize John 1:1.
    reason_for_decision: The user chose option 1 from the numbered menu, which maps to summarizing John 1:1.
    message_changed: True
</assistant_response>

## Example 5

<past_conversation>
    assistant_response: Certainly! Here’s what I can do to assist with Bible translation tasks:

        1. Summarize a passage
        2. FIA process guidance
        3. Translation helps
        4. Keywords
        5. Show scripture text
        6. Read aloud
        7. Translate scripture
        8. Set response language

    Which of these capabilities would you like to explore?
</past_conversation>

<current_message>
    user_message: 2
</current_message>

<assistant_response>
    new_message: Guide me through the FIA process for John 1:1.
    reason_for_decision: The user selected option 2 from the numbered menu, which maps to FIA guidance for John 1:1.
    message_changed: True
</assistant_response>

## Example 6

<past_conversation>
    assistant_response: Certainly! Here’s what I can do to assist with Bible translation tasks:

        1. Summarize a passage
        2. FIA process guidance
        3. Translation helps
        4. Keywords
        5. Show scripture text
        6. Read aloud
        7. Translate scripture
        8. Set response language

    Which of these capabilities would you like to explore?
</past_conversation>

<current_message>
    user_message: 3
</current_message>

<assistant_response>
    new_message: Provide translation helps for John 1:1.
    reason_for_decision: The user selected option 3 from the numbered menu, which maps to translation helps for John 1:1.
    message_changed: True
</assistant_response>

## Example 7

<past_conversation>
    assistant_response: Certainly! Here’s what I can do to assist with Bible translation tasks:

        1. Summarize a passage
        2. FIA process guidance
        3. Translation helps
        4. Keywords
        5. Show scripture text
        6. Read aloud
        7. Translate scripture
        8. Set response language

    Which of these capabilities would you like to explore?
</past_conversation>

<current_message>
    user_message: 4
</current_message>

<assistant_response>
    new_message: List the key terms in John 1:1.
    reason_for_decision: The user selected option 4 from the numbered menu, which maps to keywords for John 1:1.
    message_changed: True
</assistant_response>

## Example 8

<past_conversation>
    assistant_response: Certainly! Here’s what I can do to assist with Bible translation tasks:

        1. Summarize a passage
        2. FIA process guidance
        3. Translation helps
        4. Keywords
        5. Show scripture text
        6. Read aloud
        7. Translate scripture
        8. Set response language

    Which of these capabilities would you like to explore?
</past_conversation>

<current_message>
    user_message: 5
</current_message>

<assistant_response>
    new_message: Show me John 1:1.
    reason_for_decision: The user selected option 5 from the numbered menu, which maps to showing John 1:1.
    message_changed: True
</assistant_response>

## Example 9

<past_conversation>
    assistant_response: Certainly! Here’s what I can do to assist with Bible translation tasks:

        1. Summarize a passage
        2. FIA process guidance
        3. Translation helps
        4. Keywords
        5. Show scripture text
        6. Read aloud
        7. Translate scripture
        8. Set response language

    Which of these capabilities would you like to explore?
</past_conversation>

<current_message>
    user_message: 6
</current_message>

<assistant_response>
    new_message: I want to listen to John 1:1.
    reason_for_decision: The user selected option 6 from the numbered menu, which maps to listening to John 1:1.
    message_changed: True
</assistant_response>

## Example 10

<past_conversation>
    assistant_response: Certainly! Here’s what I can do to assist with Bible translation tasks:

        1. Summarize a passage
        2. FIA process guidance
        3. Translation helps
        4. Keywords
        5. Show scripture text
        6. Read aloud
        7. Translate scripture
        8. Set response language

    Which of these capabilities would you like to explore?
</past_conversation>

<current_message>
    user_message: 7
</current_message>

<assistant_response>
    new_message: Translate John 1:1 into Spanish.
    reason_for_decision: The user selected option 7 from the numbered menu, so we translate John 1:1 into Spanish.
    message_changed: True
</assistant_response>

## Example 11

<past_conversation>
    assistant_response: Certainly! Here’s what I can do to assist with Bible translation tasks:

        1. Summarize a passage
        2. FIA process guidance
        3. Translation helps
        4. Keywords
        5. Show scripture text
        6. Read aloud
        7. Translate scripture
        8. Set response language

    Which of these capabilities would you like to explore?
</past_conversation>

<current_message>
    user_message: 8
</current_message>

<assistant_response>
    new_message: Set my response language to Spanish.
    reason_for_decision: The user selected option 8 from the numbered menu, which maps to setting the response language to Spanish.
    message_changed: True
</assistant_response>

## Example 12

<past_conversation>
    assistant_response: Certainly! Here’s what I can do to assist with Bible translation tasks:

        1. Summarize a passage
        2. FIA process guidance
        3. Translation helps
        4. Keywords
        5. Show scripture text
        6. Read aloud
        7. Translate scripture
        8. Set response language

    Which of these capabilities would you like to explore?
</past_conversation>

<current_message>
    user_message: what can you do?
</current_message>

<assistant_response>
    new_message: what can you do?
    reason_for_decision: The user is repeating the capabilities question verbatim; the intent is already clear, so no changes were needed.
    message_changed: False
</assistant_response>
"""

INTENT_CLASSIFICATION_AGENT_SYSTEM_PROMPT = """
You are a node in a chatbot system called "BT Servant", which provides intelligent assistance to Bible translators. Your
job is to classify the intent(s) of the user's latest message. Always return at least one intent from the approved list.
If more than one intent fits, return all of them. When in doubt, fall back to `perform-unsupported-function`. If the
request is outside the scope of Bible translation, the Bible translation process, or the resources stored in the system
(for example: Translation Notes, Translation Words, FIA resources, commentaries, Bible dictionaries, etc.), return
`perform-unsupported-function`.

If the user is clearly asking for information about the BT Servant system itself—especially short cues like "help",
"help!", "help me", or "help please"—classify the message as `retrieve-system-information`.

You must choose one or more intents from the following list:

<intents>
  <intent name="get-bible-translation-assistance">
    The user is asking for help with Bible translation, including understanding meaning; finding source verses;
    clarifying language issues; consulting translation resources (Translation Notes, Bible dictionaries, commentaries,
    translation words, etc.); requesting explanations of resource content; asking for summaries or transformations of
    resource content; or asking questions about scripture that support translation work. If the user specifically wants
    FIA guidance, use `consult-fia-resources`.
  </intent>
  <intent name="consult-fia-resources">
    The user is asking about the Familiarization, Internalization, and Articulation (FIA) process: its steps, how to
    practice it, or how to apply it to a passage or team. Choose this intent whenever FIA guidance or FIA resources are
    the focus, even if a passage is mentioned.
  </intent>
  <intent name="get-passage-summary">
    The user explicitly wants a summary of a Bible passage, verse range, chapter(s), or entire book (for example:
    "Summarize John 3:16-18", "Summarize John 1-4", "Summarize John"). If multiple books are mentioned, still return
    this intent—downstream logic will handle scope.
  </intent>
  <intent name="get-passage-keywords">
    The user explicitly wants key words from a specific passage, verse range, chapter(s), or book (for example:
    "What are the keywords in Genesis 1?", "List key words in John 3:16"). Only use this intent when a Scripture
    selection is present.
  </intent>
  <intent name="get-translation-helps">
    The user wants translation challenges, considerations, guidance, or alternate renderings for a passage or book.
    Examples: "Help me translate Titus 1:1-5", "What are translation challenges in Exodus 1?", "Alternate translations
    for 'only begotten' in John 3:16".
  </intent>
  <intent name="retrieve-scripture">
    The user wants the exact Bible text (verbatim verses), possibly in a specific language or version (for example:
    "Give me John 1:1 in Indonesian", "Provide the text of Job 1:1-5"). Use this when they want the text itself.
  </intent>
  <intent name="listen-to-scripture">
    The user wants scripture read aloud (audio output) for a specific passage, verse range, chapter(s), or book. This is
    the audio equivalent of `retrieve-scripture`. Examples: "Read John 3 out loud", "Let me listen to John 1:1-5",
    "Play Genesis 1 in Spanish".
  </intent>
  <intent name="translate-scripture">
    The user wants the scripture text translated into another language (for example: "Translate John 1:1 into Portuguese",
    "Translate the French version of John 1:1 into Spanish").
  </intent>
  <intent name="set-response-language">
    The user wants to change the assistant's response language (for example: "Respond in Spanish", "Use Portuguese").
  </intent>
  <intent name="set-agentic-strength">
    The user wants to change the agentic strength of the assistant's responses (for example: "Set my agentic strength to
    low", "Increase the detail of your answers"). Supported levels: normal, low, very_low.
  </intent>
  <intent name="retrieve-system-information">
    The user wants information about the BT Servant system itself—its resources, capabilities, uptime, data sources, or
    other operational details.
  </intent>
  <intent name="perform-unsupported-function">
    Use this when the user's request does not match any other intent or is outside the system's documented capabilities.
    Examples include jokes, timers, internet searches, or broad corpus queries such as "Find every verse with <word>".
  </intent>
  <intent name="converse-with-bt-servant">
    The user is trying to have a general conversation or casual chat unrelated to Bible translation (for example:
    greetings, "How are you?", "What's up?").
  </intent>
</intents>

Here are example classifications:

<examples>
  <example>
    <message>tell me about ephesus</message>
    <intent>get-bible-translation-assistance</intent>
  </example>
  <example>
    <message>What is a denarius?</message>
    <intent>get-bible-translation-assistance</intent>
  </example>
  <example>
    <message>What is the fourth step of the FIA process?</message>
    <intent>consult-fia-resources</intent>
  </example>
  <example>
    <message>Explain the FIA process to me like I'm a three year old.</message>
    <intent>consult-fia-resources</intent>
  </example>
  <example>
    <message>Summarize Mark 3.</message>
    <intent>get-passage-summary</intent>
  </example>
  <example>
    <message>Summarize Titus 3:4.</message>
    <intent>get-passage-summary</intent>
  </example>
  <example>
    <message>keywords in Genesis and Exodus</message>
    <intent>get-passage-keywords</intent>
  </example>
  <example>
    <message>List the keywords for Gen 1-3.</message>
    <intent>get-passage-keywords</intent>
  </example>
  <example>
    <message>What are alternate translations for "only begotten" in John 3:16?</message>
    <intent>get-translation-helps</intent>
  </example>
  <example>
    <message>Help me translate Titus 1:1-5.</message>
    <intent>get-translation-helps</intent>
  </example>
  <example>
    <message>translate John 1:1 into Portuguese</message>
    <intent>translate-scripture</intent>
  </example>
  <example>
    <message>translate the French version of John 1:1 into Spanish</message>
    <intent>translate-scripture</intent>
  </example>
  <example>
    <message>Please provide the text of Job 1:1-5.</message>
    <intent>retrieve-scripture</intent>
  </example>
  <example>
    <message>Can you give me John 1:1 from the Indonesian Bible?</message>
    <intent>retrieve-scripture</intent>
  </example>
  <example>
    <message>Read John 3 out loud.</message>
    <intent>listen-to-scripture</intent>
  </example>
  <example>
    <message>Let me listen to John 1:1-5.</message>
    <intent>listen-to-scripture</intent>
  </example>
  <example>
    <message>Play Genesis 1 in Spanish.</message>
    <intent>listen-to-scripture</intent>
  </example>
  <example>
    <message>Can you reply to me in French from now on?</message>
    <intent>set-response-language</intent>
  </example>
  <example>
    <message>Set my agentic strength to low.</message>
    <intent>set-agentic-strength</intent>
  </example>
  <example>
    <message>Where does BT Servant get its information from?</message>
    <intent>retrieve-system-information</intent>
  </example>
  <example>
    <message>help!</message>
    <intent>retrieve-system-information</intent>
  </example>
  <example>
    <message>What else can you do?</message>
    <intent>retrieve-system-information</intent>
  </example>
  <example>
    <message>Can you tell me a joke?</message>
    <intent>perform-unsupported-function</intent>
  </example>
  <example>
    <message>hello</message>
    <intent>converse-with-bt-servant</intent>
  </example>
  <example>
    <message>How are you doing today?</message>
    <intent>converse-with-bt-servant</intent>
  </example>
</examples>

Return a single JSON object of the form:
```json
{ "intents": ["get-bible-translation-assistance"] }
```
"""

INTENT_CLASSIFICATION_STRUCTURED_PROMPT = """
You are a node in a chatbot system called "BT Servant", which provides intelligent assistance to Bible translators. Your
job is to classify the intent(s) of the user's latest message AND extract the relevant parameters for each intent.

This is crucial for multi-intent messages where the user requests multiple things. For example, if they say "Summarize
Romans 8 and translate it to Spanish", you need to identify both intents AND extract the parameters:
- get-passage-summary with parameters_json: '{"passage": "Romans 8"}'
- translate-scripture with parameters_json: '{"passage": "Romans 8", "target_language": "Spanish"}'

# Instructions

1. Always return at least one intent from the approved list
2. If more than one intent fits, return ALL of them with their specific parameters
3. Extract parameters relevant to each intent (passage references, languages, etc.)
4. Resolve pronouns and references ("it" → the last mentioned passage)
5. When in doubt, fall back to `perform-unsupported-function`
6. If the request is outside Bible translation scope, return `perform-unsupported-function`
7. For "help" requests, classify as `retrieve-system-information`

# Parameter Schemas by Intent

- **get-passage-summary**: `{"passage": "string"}` - The passage to summarize (e.g., "Romans 8", "John 3:16-18")
- **get-passage-keywords**: `{"passage": "string"}` - The passage to extract keywords from
- **get-translation-helps**: `{"passage": "string"}` - The passage needing translation help
- **retrieve-scripture**: `{"passage": "string", "source_language"?: "string"}` - Passage and optional source language
- **listen-to-scripture**: `{"passage": "string", "source_language"?: "string"}` - Passage for audio playback
- **translate-scripture**: `{"passage": "string", "target_language"?: "string", "source_language"?: "string"}`
- **set-response-language**: `{"language": "string"}` - The desired response language
- **set-agentic-strength**: `{"strength": "string"}` - One of: "normal", "low", "very_low"
- **consult-fia-resources**: `{"passage"?: "string", "topic"?: "string"}` - Optional passage or FIA topic
- **get-bible-translation-assistance**: `{"query": "string"}` - The general question or topic
- **converse-with-bt-servant**: `{}` - No parameters needed for greetings/chitchat
- **retrieve-system-information**: `{}` - No parameters
- **perform-unsupported-function**: `{}` - No parameters

# Examples

## Example 1: Single Intent
<message>Summarize Romans 8</message>
<classification>
{
  "intents": [
    {
      "intent": "get-passage-summary",
      "parameters_json": "{\\"passage\\": \\"Romans 8\\"}"
    }
  ]
}
</classification>

## Example 2: Multiple Intents
<message>Summarize Romans 8 and translate it to Spanish</message>
<classification>
{
  "intents": [
    {
      "intent": "get-passage-summary",
      "parameters_json": "{\\"passage\\": \\"Romans 8\\"}"
    },
    {
      "intent": "translate-scripture",
      "parameters_json": "{\\"passage\\": \\"Romans 8\\", \\"target_language\\": \\"Spanish\\"}"
    }
  ]
}
</classification>

## Example 3: Pronoun Resolution
<message>Show me John 3:16 and then summarize it</message>
<classification>
{
  "intents": [
    {
      "intent": "retrieve-scripture",
      "parameters_json": "{\\"passage\\": \\"John 3:16\\"}"
    },
    {
      "intent": "get-passage-summary",
      "parameters_json": "{\\"passage\\": \\"John 3:16\\"}"
    }
  ]
}
</classification>

## Example 4: Multiple Passages
<message>Summarize Romans 8, translate John 1:1 into French, and give me keywords for Mark 3</message>
<classification>
{
  "intents": [
    {
      "intent": "get-passage-summary",
      "parameters_json": "{\\"passage\\": \\"Romans 8\\"}"
    },
    {
      "intent": "translate-scripture",
      "parameters_json": "{\\"passage\\": \\"John 1:1\\", \\"target_language\\": \\"French\\"}"
    },
    {
      "intent": "get-passage-keywords",
      "parameters_json": "{\\"passage\\": \\"Mark 3\\"}"
    }
  ]
}
</classification>

## Example 5: FIA with Passage
<message>Walk me through the FIA process for Titus 1:1-5</message>
<classification>
{
  "intents": [
    {
      "intent": "consult-fia-resources",
      "parameters_json": "{\\"passage\\": \\"Titus 1:1-5\\"}"
    }
  ]
}
</classification>

## Example 6: Settings
<message>Set my response language to Indonesian</message>
<classification>
{
  "intents": [
    {
      "intent": "set-response-language",
      "parameters_json": "{\\"language\\": \\"Indonesian\\"}"
    }
  ]
}
</classification>

Return a JSON object matching the IntentWithContext schema. Be thorough in extracting all parameters.
"""

DETECT_LANGUAGE_AGENT_SYSTEM_PROMPT = """
You are a language detection specialist for a Bible translation assistant chatbot.

# Task
Detect the ISO 639-1 language code of the provided text. Consider the domain context: users often mention Bible book
names (e.g., "Dan" for Daniel) and may mix English instructions with non-English queries.

# Instructions
- Return the two-letter ISO 639-1 code (e.g., "en", "es", "fr", "id", etc.)
- When ambiguous (e.g., "Dan" could be Indonesian "dan" or the book "Daniel"), prefer English if the text contains
  typical Bible references or English instruction words
- If you cannot confidently detect the language, return "en" as the fallback

# Examples
- "Summarize Dan 1:1" → "en" (Bible reference)
- "Terjemahkan Roma 1:1" → "id" (Indonesian)
- "Résumer Jean 3:16" → "fr" (French)
- "keywords in John 3" → "en" (English instruction)
"""


# ========== SCHEMA CLASSES ==========


class PreprocessorResult(BaseModel):
    """Result type for the preprocessor node output."""

    new_message: str
    reason_for_decision: str
    message_changed: bool


# ========== HELPER FUNCTIONS ==========


def resolve_agentic_strength(state: dict[str, Any]) -> str:
    """Return the effective agentic strength, honoring user overrides when set."""
    candidate = cast(
        Optional[str], state.get("agentic_strength") or state.get("user_agentic_strength")
    )
    if isinstance(candidate, str):
        lowered = candidate.lower()
        if lowered in ALLOWED_AGENTIC_STRENGTH:
            return lowered

    configured = getattr(config, "AGENTIC_STRENGTH", "normal")
    if isinstance(configured, str):
        configured_lower = configured.lower()
        if configured_lower in ALLOWED_AGENTIC_STRENGTH:
            return configured_lower
    return "normal"


def model_for_agentic_strength(
    agentic_strength: str,
    *,
    allow_low: bool,
    allow_very_low: bool,
) -> str:
    """Return GPT model name based on strength and allowed downgrades."""
    allowed: set[str] = set()
    if allow_low:
        allowed.add("low")
    if allow_very_low:
        allowed.add("very_low")
    return "gpt-4o-mini" if agentic_strength in allowed else "gpt-4o"


# ========== MAIN FUNCTIONS ==========


def detect_language(client: OpenAI, text: str, *, agentic_strength: Optional[str] = None) -> str:  # pylint: disable=too-many-locals
    """Detect ISO 639-1 language code of the given text via OpenAI.

    Uses a domain-aware prompt with deterministic decoding and a light
    heuristic to avoid false Indonesian due to Bible abbreviations like
    "Dan" (Daniel).
    """
    messages: list[EasyInputMessageParam] = [
        {
            "role": "user",
            "content": f"text: {text}",
        },
    ]
    strength_source = (
        agentic_strength
        if agentic_strength is not None
        else getattr(config, "AGENTIC_STRENGTH", "normal")
    )
    strength = str(strength_source).lower()
    if strength not in ALLOWED_AGENTIC_STRENGTH:
        strength = "normal"
    model_name = model_for_agentic_strength(strength, allow_low=True, allow_very_low=True)
    response = client.responses.parse(
        model="gpt-4o",
        instructions=DETECT_LANGUAGE_AGENT_SYSTEM_PROMPT,
        input=cast(Any, messages),
        text_format=MessageLanguage,
        temperature=0,
        store=False,
    )
    usage = getattr(response, "usage", None)
    track_openai_usage(usage, model_name, _extract_cached_input_tokens, add_tokens)
    message_language = cast(MessageLanguage | None, response.output_parsed)
    predicted = message_language.language.value if message_language else "en"
    logger.info("language detection (model): %s", predicted)

    # Heuristic guard: If we predicted Indonesian ('id') but the text looks like
    # an English instruction paired with a Bible reference, prefer English.
    # This specifically addresses the common "Dan" (Daniel) vs Indonesian "dan" ambiguity.
    try:
        has_english_instruction = bool(
            re.search(
                r"\b(summarize|explain|what|who|why|how|list|give|provide)\b",
                str(text),
                re.IGNORECASE,
            )
        )
        has_verse_pattern = bool(re.search(r"\b[A-Za-z]{2,4}\s+\d+:\d+\b", str(text)))
        logger.info(
            "heuristic_guard: predicted=%s english_instruction=%s verse_pattern=%s",
            predicted,
            has_english_instruction,
            has_verse_pattern,
        )
        if predicted == "id" and has_english_instruction and has_verse_pattern:
            logger.info(
                "heuristic_guard: overriding id -> en due to English instruction + verse pattern"
            )
            predicted = "en"
    except re.error as err:
        # If regex fails for any reason, fall back to the model prediction.
        logger.info(
            "heuristic_guard: regex error (%s); keeping model prediction: %s", err, predicted
        )

    return predicted


def determine_query_language(
    client: OpenAI, query: str, agentic_strength: str
) -> tuple[str, list[str]]:
    """Determine the language of the user's original query and set collection order.

    Returns:
        Tuple of (query_language, stack_rank_collections)
    """
    query_language = detect_language(client, query, agentic_strength=agentic_strength)
    logger.info("language code %s detected by gpt-4o.", query_language)
    stack_rank_collections = [
        "knowledgebase",
        "en_resources",
    ]
    # If the detected language is not English, also search the matching
    # language-specific resources collection (e.g., "es_resources").
    if query_language and query_language != "en" and query_language != Language.OTHER.value:
        localized_collection = f"{query_language}_resources"
        stack_rank_collections.append(localized_collection)
        logger.info(
            "appended localized resources collection: %s (language=%s)",
            localized_collection,
            query_language,
        )

    return query_language, stack_rank_collections


def determine_intents(client: OpenAI, query: str) -> list[IntentType]:
    """Classify the user's transformed query into one or more intents."""
    messages: list[EasyInputMessageParam] = [
        {
            "role": "system",
            "content": INTENT_CLASSIFICATION_AGENT_SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": f"what is your classification of the latest user message: {query}",
        },
    ]
    response = client.responses.parse(
        model="gpt-4o", input=cast(Any, messages), text_format=UserIntents, store=False
    )
    usage = getattr(response, "usage", None)
    track_openai_usage(usage, "gpt-4o", _extract_cached_input_tokens, add_tokens)
    user_intents_model = cast(UserIntents, response.output_parsed)
    logger.info(
        "extracted user intents: %s", " ".join([i.value for i in user_intents_model.intents])
    )

    return user_intents_model.intents


def determine_intents_structured(client: OpenAI, query: str) -> list[IntentWithContext]:
    """Classify the user's query into intents WITH extracted parameters.

    This enhanced version performs structured intent detection, extracting relevant
    parameters for each intent at classification time. This is crucial for multi-intent
    messages where we need to disambiguate which parts of the message apply to which intent.

    For example:
    - "Summarize Romans 8 and translate it to Spanish"
      Returns:
        [IntentWithContext(intent=GET_PASSAGE_SUMMARY, parameters={"passage": "Romans 8"}),
         IntentWithContext(intent=TRANSLATE_SCRIPTURE, parameters={"passage": "Romans 8", "target_language": "Spanish"})]

    Args:
        client: OpenAI client instance
        query: The user's (possibly preprocessed) query text

    Returns:
        List of IntentWithContext objects with pre-extracted parameters
    """
    logger.info(
        "[intent-detection-structured] Classifying query with parameter extraction: %s", query[:100]
    )

    messages: list[EasyInputMessageParam] = [
        {
            "role": "system",
            "content": INTENT_CLASSIFICATION_STRUCTURED_PROMPT,
        },
        {
            "role": "user",
            "content": f"Classify this message and extract parameters: {query}",
        },
    ]

    try:
        response = client.responses.parse(
            model="gpt-4o",
            input=cast(Any, messages),
            text_format=UserIntentsStructured,
            store=False,
        )
        usage = getattr(response, "usage", None)
        track_openai_usage(usage, "gpt-4o", _extract_cached_input_tokens, add_tokens)

        structured_intents = cast(UserIntentsStructured, response.output_parsed)

        # Log extracted intents with parameters
        logger.info(
            "[intent-detection-structured] Detected %d intent(s):",
            len(structured_intents.intents),
        )
        for idx, intent_ctx in enumerate(structured_intents.intents):
            logger.info(
                "[intent-detection-structured]   Intent %d: %s with params=%s",
                idx + 1,
                intent_ctx.intent.value,
                intent_ctx.parameters,
            )

        return structured_intents.intents

    except Exception:  # pylint: disable=broad-except
        logger.error(
            "[intent-detection-structured] Structured detection failed, falling back to simple detection",
            exc_info=True,
        )
        # Fallback to simple detection if structured fails
        simple_intents = determine_intents(client, query)
        logger.warning(
            "[intent-detection-structured] Fell back to simple detection, got %d intent(s): %s",
            len(simple_intents),
            [i.value for i in simple_intents],
        )
        # Convert to IntentWithContext with empty parameters
        return [IntentWithContext(intent=intent, parameters_json="{}") for intent in simple_intents]


def preprocess_user_query(
    client: OpenAI, query: str, chat_history: list[dict[str, str]]
) -> tuple[str, str, bool]:  # pylint: disable=too-many-locals
    """Lightly clarify or correct the user's query using conversation history.

    Returns:
        Tuple of (transformed_query, reason_for_decision, message_changed)
    """
    history_context_message = f"past_conversation: {json.dumps(chat_history)}"
    messages: list[EasyInputMessageParam] = [
        {
            "role": "user",
            "content": history_context_message,
        },
        {
            "role": "user",
            "content": f"current_message: {query}",
        },
    ]
    response = client.responses.parse(
        model="gpt-4o",
        instructions=PREPROCESSOR_AGENT_SYSTEM_PROMPT,
        input=cast(Any, messages),
        text_format=PreprocessorResult,
        store=False,
    )
    usage = getattr(response, "usage", None)
    track_openai_usage(usage, "gpt-4o", _extract_cached_input_tokens, add_tokens)
    preprocessor_result = cast(PreprocessorResult | None, response.output_parsed)
    if preprocessor_result is None:
        new_message = query
        reason_for_decision = "no changes"
        message_changed = False
    else:
        new_message = preprocessor_result.new_message
        reason_for_decision = preprocessor_result.reason_for_decision
        message_changed = preprocessor_result.message_changed
    logger.info(
        "original_message: %s\nnew_message: %s\nreason_for_decision: %s\nmessage_changed: %s",
        query,
        new_message,
        reason_for_decision,
        message_changed,
    )
    transformed_query = new_message if message_changed else query
    return transformed_query, reason_for_decision, message_changed


def is_affirmative_response_to_continuation(
    client: OpenAI, user_message: str, continuation_context: str
) -> bool:
    """Use LLM to determine if user is responding affirmatively to a continuation prompt.

    Args:
        client: OpenAI client
        user_message: The user's response message
        continuation_context: Description of what we asked (e.g., "translate Romans 8 to Spanish")

    Returns:
        True if the user is responding affirmatively, False otherwise

    Examples:
        - is_affirmative("yes", "translate...") -> True
        - is_affirmative("sure", "translate...") -> True
        - is_affirmative("no", "translate...") -> False
        - is_affirmative("what's Romans 9 about?", "translate...") -> False (pivoted)
    """
    logger.info(
        "[affirmative-detection] Checking if '%s' is affirmative to: %s",
        user_message[:50] + "..." if len(user_message) > 50 else user_message,
        continuation_context[:50] + "..."
        if len(continuation_context) > 50
        else continuation_context,
    )

    prompt = f"""You are analyzing whether a user is responding affirmatively to a continuation question.

Context: We asked the user if they wanted us to: "{continuation_context}"

User's response: "{user_message}"

Determine if the user is saying YES (affirmative) or NO/SOMETHING_ELSE.

Return "YES" if the user is clearly agreeing, confirming, or saying yes.
Return "NO" if:
- They explicitly decline (no, nope, cancel, skip, etc.)
- They ask a different question (pivot to new topic)
- They ignore the continuation and make a new request
- They respond with anything other than clear agreement

Examples:
- "yes" -> YES
- "sure" -> YES
- "ok" -> YES
- "please" -> YES
- "go ahead" -> YES
- "no" -> NO
- "not now" -> NO
- "what about Romans 9?" -> NO (new question)
- "summarize John 3" -> NO (new request)
- "tell me about David" -> NO (different topic)

Respond with exactly one word: YES or NO"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=10,
            store=False,
        )

        usage = getattr(response, "usage", None)
        track_openai_usage(usage, "gpt-4o-mini", _extract_cached_input_tokens, add_tokens)

        content = response.choices[0].message.content
        if not content:
            logger.warning(
                "[affirmative-detection] LLM returned empty content, defaulting to False"
            )
            return False

        answer = content.strip().upper()
        is_affirmative = answer == "YES"

        logger.info(
            "[affirmative-detection] LLM returned '%s', interpreted as: %s",
            answer,
            "AFFIRMATIVE" if is_affirmative else "NOT AFFIRMATIVE",
        )

        return is_affirmative

    except Exception:  # pylint: disable=broad-except
        logger.error(
            "[affirmative-detection] Error during LLM call, defaulting to False (conservative)",
            exc_info=True,
        )
        return False


__all__ = [
    "PREPROCESSOR_AGENT_SYSTEM_PROMPT",
    "INTENT_CLASSIFICATION_AGENT_SYSTEM_PROMPT",
    "INTENT_CLASSIFICATION_STRUCTURED_PROMPT",
    "DETECT_LANGUAGE_AGENT_SYSTEM_PROMPT",
    "PreprocessorResult",
    "resolve_agentic_strength",
    "model_for_agentic_strength",
    "detect_language",
    "determine_query_language",
    "determine_intents",
    "determine_intents_structured",
    "is_affirmative_response_to_continuation",
    "preprocess_user_query",
]
