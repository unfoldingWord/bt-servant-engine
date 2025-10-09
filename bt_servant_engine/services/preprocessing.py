"""Query preprocessing, language detection, and intent classification."""

from __future__ import annotations

import json
import re
from typing import Any, Optional, cast

from openai import OpenAI
from openai.types.responses.easy_input_message_param import EasyInputMessageParam
from pydantic import BaseModel

from bt_servant_engine.core.agentic import ALLOWED_AGENTIC_STRENGTH
from bt_servant_engine.core.config import config
from bt_servant_engine.core.intents import IntentType, UserIntents
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
as little as possible. Change nothing unless necessary. If the intent of the user's message is already clear,
change nothing. Never greatly expand the user's current message. Changes should be small or none. Feel free to fix
obvious spelling mistakes or errors, but not logic errors like incorrect books of the Bible. Do NOT narrow the scope of
explicit scripture selections: if a user requests multiple chapters, verse ranges, or disjoint selections (including
conjunctions like "and" or comma/semicolon lists), preserve them exactly as written. If the system has constraints
(for example, only a single chapter can be processed at a time), do NOT modify the user's message to fit those
constraints — leave the message intact and let downstream nodes handle any rejection or guidance. For translation
requests, do NOT add or change a target language; preserve only what the user explicitly stated.
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
"""

INTENT_CLASSIFICATION_AGENT_SYSTEM_PROMPT = """
You are a node in a chatbot system called "BT Servant", which provides intelligent assistance to Bible translators. Your
job is to classify the **intent(s)** of the user's latest message. Always return **at least one** intent from the
approved list. However, if more than one intent is found, make sure to return those as well. If you're unsure, return
`perform-unsupported-function`. If the user is asking for something outside the scope of the Bible, Bible translation,
the Bible translation process, or one of the resources stored in the system (ex. Translation Notes, FIA resources,
the Bible, Translation Words, Greek or Hebrew resources, commentaries, Bible dictionaries, etc.), or something outside
system capabilities (defined by the various intents), also return the `perform-unsupported-function` intent.

You MUST always return at least one intent. You MUST choose one or more intents from the following intent types:

<intents>
  <intent name="get-bible-translation-assistance">
    The user is asking for help with Bible translation — including understanding meaning; finding source verses;
    clarifying language issues; consulting translation resources (ex. Translation Notes, the Bible, translation words,
    commentaries, etc); receiving explanation of resources; interacting with resource content; asking for
    summarizations or transformations of resource content (translate resource content to French, simplify it for
    children, list all verbs, etc.); asking questions about scripture (What is the main point of Romans?), or anything
    to do with actually understanding and translating the Bible itself. Use this intent when the user wants to use the
    Bible study and translation resources.
  </intent>

  <intent name="get-passage-summary">
    The user wants a summary of a specific passage or book. For example: "Summarize Romans", "Summarize Matthew 5",
    "Summarize Matthew 5:3-8". Use this intent when there is an explicit, Bible passage citation or reference and the
    request is specifically to summarize that passage.
  </intent>

  <intent name="get-passage-keywords">
    The user wants the main keywords found in a specific passage. A complete Bible passage citation or reference is
    always required for this intent. For example: "keywords in John 3:16", "list keywords in 1 Cor", "main words in
    Romans 1". Do NOT use if no scripture selection is present.
  </intent>

  <intent name="get-passage-audio">
    The user wants to listen to a specific passage from the Bible. A complete Bible passage citation or reference is
    always required for this intent. For example: "Listen to John 1:1", "read Mark 2", "play John 3:16". Do NOT use if
    no scripture selection is present.
  </intent>

  <intent name="get-translation-helps">
    The user is asking for translation help, challenges, considerations, guidance, or alternate renderings for a given
    passage or book. This includes asking about translation options, alternate translations, or what to consider when
    translating specific verses. A complete Bible passage citation or reference is always required for this intent.
    Examples: "Help me translate Titus 1:1-5", "translation challenges for Exo 1", "what to consider when translating
    Ruth", "give me translation help for John 3:16", "get translation notes for Mark 2:4-5".
    Do NOT use if no scripture selection is present.
  </intent>

  <intent name="consult-fia-resources">
    The user is asking about the Familiarization, Internalization, and Articulation (FIA) process, its steps, or how to
    apply those steps to a passage, team, or translation scenario. Examples include learning the FIA workflow, asking
    what a particular step looks like, how to translate the Bible using FIA, or how FIA should be practiced in a specific
    chapter. Choose this intent whenever FIA guidance or FIA resources are the focus, even if a passage is mentioned.
    Examples: "How do I translate the Bible?", "What are the steps of the FIA process?", "What does FIA step 2 look like
    in the first chapter of Mark?", "what is FIA?", "How do I apply FIA?".
  </intent>

  <intent name="set-response-language">
    The user wants to set or change the language for the system's responses. Examples: "respond in Spanish",
    "switch to French", "use Portuguese", etc.
  </intent>

  <intent name="set-agentic-strength">
    The user is trying to adjust the system's "agentic strength" — how detailed or sophisticated the AI's responses
    should be. Examples: "lower agentic strength", "set strength to high", "increase detail", etc.
  </intent>

  <intent name="get-help">
    The user is asking for help using the system, wants to know what the system can do, or is asking for
    documentation/instructions. Examples: "help", "what can you do?", "how do I use this?", etc.
  </intent>

  <intent name="perform-unsupported-function">
    Use this when the user's request doesn't match any of the above intents, or when the user is asking for something
    outside the system's documented capabilities. Also use this when you're unsure of the correct intent.
  </intent>

  <intent name="converse">
    The user is trying to have a general conversation, asking a personal question, or just chatting in a way that
    doesn't relate to Bible translation. Examples: "hello", "how are you?", "what's the weather?", etc.
  </intent>
</intents>
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
        "new_message: %s\nreason_for_decision: %s\nmessage_changed: %s",
        new_message,
        reason_for_decision,
        message_changed,
    )
    transformed_query = new_message if message_changed else query
    return transformed_query, reason_for_decision, message_changed


__all__ = [
    "PREPROCESSOR_AGENT_SYSTEM_PROMPT",
    "INTENT_CLASSIFICATION_AGENT_SYSTEM_PROMPT",
    "DETECT_LANGUAGE_AGENT_SYSTEM_PROMPT",
    "PreprocessorResult",
    "resolve_agentic_strength",
    "model_for_agentic_strength",
    "detect_language",
    "determine_query_language",
    "determine_intents",
    "preprocess_user_query",
]
