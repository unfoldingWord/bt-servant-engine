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
obvious spelling mistakes or errors, but not logic errors like incorrect books of the Bible.

**CRITICAL: Handling Yes/No Responses**
When the user responds with simple affirmations or negations like "yes", "no", "yeah", "nope", "yes please", "no thanks",
etc., you MUST expand these using the immediately preceding assistant message (the most recent context). These short
responses almost always refer to a follow-up question or offer from the assistant. Extract the action or topic from the
previous assistant message and expand the user's response into a complete, actionable request. Pay special attention to
the VERY LAST sentence or question in the assistant's previous message, as that is typically what the user is responding to. Do NOT narrow the scope of
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

## Example 13: Expanding "yes" after a follow-up question

<past_conversation>
    user_message: Show me John 3:16
    assistant_response: In the beginning was the Word, and the Word was with God, and the Word was God.

    Would you like to look up another Bible passage?
</past_conversation>

<current_message>
    user_message: yes
</current_message>

<assistant_response>
    new_message: Yes, I would like to look up another Bible passage.
    reason_for_decision: The user is affirming the assistant's follow-up question about looking up another passage. Expanded "yes" into a complete request based on the immediately preceding question.
    message_changed: True
</assistant_response>

## Example 14: Expanding "no" after a follow-up question

<past_conversation>
    user_message: Provide translation helps for Romans 8:1
    assistant_response: [Translation helps content]

    Do you need help with another translation question?
</past_conversation>

<current_message>
    user_message: no
</current_message>

<assistant_response>
    new_message: No, I don't need help with another translation question.
    reason_for_decision: The user is declining the assistant's offer. Expanded "no" to make the intent explicit.
    message_changed: True
</assistant_response>

## Example 15: Expanding "yes please" with specific context

<past_conversation>
    user_message: Summarize Acts 2
    assistant_response: [Summary of Acts 2]

    Would you like a summary of another passage?
</past_conversation>

<current_message>
    user_message: yes please
</current_message>

<assistant_response>
    new_message: Yes, I would like a summary of another passage.
    reason_for_decision: The user is affirming the follow-up question. Expanded to clarify they want another summary.
    message_changed: True
</assistant_response>

## Example 16: Expanding "yeah" after passage suggestion

<past_conversation>
    user_message: Provide translation helps for John 3:1-5
    assistant_response: [Translation helps for John 3:1-5]

    Would you like help translating John 3:6-10 as well?
</past_conversation>

<current_message>
    user_message: yeah
</current_message>

<assistant_response>
    new_message: Yes, I would like help translating John 3:6-10.
    reason_for_decision: The user is affirming the specific suggestion about John 3:6-10. Expanded to include the specific passage mentioned.
    message_changed: True
</assistant_response>

## Example 17: Expanding "nope" after topic suggestion

<past_conversation>
    user_message: Who was Apollos in the Bible?
    assistant_response: Apollos was a learned Jew from Alexandria who became a prominent Christian teacher and evangelist in the early church. He is mentioned in Acts 18 and 1 Corinthians, where Paul describes him as an eloquent speaker who powerfully refuted Jewish objections to Christianity. Apollos worked closely with Aquila and Priscilla and was influential in Corinth.

    Would you like to explore another biblical topic?
</past_conversation>

<current_message>
    user_message: nope
</current_message>

<assistant_response>
    new_message: No, I don't want to explore another biblical topic.
    reason_for_decision: The user is declining the offer to explore another topic. Expanded to make the refusal explicit.
    message_changed: True
</assistant_response>

## Example 18: Expanding "yes" with person/topic context

<past_conversation>
    user_message: Who was Apollos in the Bible?
    assistant_response: Apollos was a learned Jew from Alexandria who became a prominent Christian teacher. He worked with Aquila, Priscilla, and Paul in the early church.

    Would you like to know more about Paul?
</past_conversation>

<current_message>
    user_message: yes
</current_message>

<assistant_response>
    new_message: Yes, I would like to know more about Paul.
    reason_for_decision: The user is affirming interest in learning about Paul. Expanded to include the specific person mentioned in the follow-up.
    message_changed: True
</assistant_response>

## Example 19: Not expanding "yes" when context is unclear

<past_conversation>
    user_message: What does "justification" mean in Romans 3?
    assistant_response: Justification in Romans 3 refers to God declaring sinners righteous through faith in Jesus Christ, not through works of the law.
</past_conversation>

<current_message>
    user_message: yes
</current_message>

<assistant_response>
    new_message: yes
    reason_for_decision: The previous assistant message did not end with a question or offer, so the meaning of "yes" is unclear. Left unchanged per instructions to avoid speculation.
    message_changed: False
</assistant_response>

## Example 20: Expanding "no thanks" after specific offer

<past_conversation>
    user_message: List the key terms in Matthew 5:1-10
    assistant_response: [Key terms listed]

    Would you like keywords from another passage?
</past_conversation>

<current_message>
    user_message: no thanks
</current_message>

<assistant_response>
    new_message: No, I don't want keywords from another passage.
    reason_for_decision: The user is politely declining the offer for more keywords. Expanded to make the refusal clear.
    message_changed: True
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

CONTINUATION_ACTION_GENERATION_PROMPT = """
You are generating complete continuation questions for a multi-intent query in the BT Servant chatbot system.

Given a user query and a list of detected intents, generate a complete, grammatically correct continuation question for each intent in the target language.

# Rules
- Generate COMPLETE questions, not fragments (e.g., "Would you like me to show key terms from Gen 4:2?")
- Each question must be grammatically correct and natural in the target language
- Include specific context from the query (passage references, names, topics)
- Use polite, conversational tone appropriate for the target language
- Keep questions concise (under 20 words)
- Generate in the target language specified (ISO 639-1 code: {target_language})
- DO NOT translate word-by-word from English - use natural phrasing for that language

# Intent Types
- get-passage-keywords: Extract key terms from a passage
- get-passage-summary: Summarize a passage
- get-translation-helps: Provide translation help for a passage
- get-bible-translation-assistance: Answer general Bible translation questions about topics, names, concepts
- consult-fia-resources: Consult FIA (Familiarization, Internalization, Articulation) resources
- retrieve-scripture: Retrieve the text of a passage
- translate-scripture: Translate a passage to another language
- listen-to-scripture: Read a passage aloud

# Examples (English)

Query: "Tell me about Barnabas, keywords in Gen 4:2, help translating John 1:4"
Intents: ["get-bible-translation-assistance", "get-passage-keywords", "get-translation-helps"]
Questions:
["Would you like me to tell you about Barnabas in the Bible?", "Would you like me to show key terms from Gen 4:2?", "Would you like me to provide translation helps for John 1:4?"]

Query: "Summarize Romans 8 and translate it to Spanish"
Intents: ["get-passage-summary", "translate-scripture"]
Questions:
["Would you like me to summarize Romans 8?", "Would you like me to translate Romans 8 to Spanish?"]

Query: "What does Ephesus mean and give me keywords in Acts 19"
Intents: ["get-bible-translation-assistance", "get-passage-keywords"]
Questions:
["Would you like me to tell you about Ephesus?", "Would you like me to show key terms from Acts 19?"]

# Examples (French - note natural phrasing, not word-for-word translation)

Query: "Parle-moi de Barnabas et donne-moi les mots-clés de Gen 4:2"
Intents: ["get-bible-translation-assistance", "get-passage-keywords"]
Questions:
["Voulez-vous que je vous parle de Barnabas dans la Bible?", "Voulez-vous que je vous montre les termes clés de Gen 4:2?"]

Now generate complete continuation questions for this query:

Query: {query}
Intents: {intents}
Target Language: {target_language}

Return ONLY a JSON array of complete question strings, one per intent, in the same order as the intents list.
Example format: ["Question 1?", "Question 2?", "Question 3?"]
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


def determine_intents_structured(
    client: OpenAI, query: str, expected_intents: list[IntentType]
) -> list[IntentWithContext]:
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
        "[intent-detection-structured] Extracting context snippets for %d pre-classified intent(s): %s",
        len(expected_intents),
        [intent.value for intent in expected_intents],
    )

    if not expected_intents:
        logger.warning(
            "[intent-detection-structured] No expected intents provided; skipping structured extraction"
        )
        return []

    messages: list[EasyInputMessageParam] = [
        {
            "role": "system",
            "content": INTENT_CLASSIFICATION_STRUCTURED_PROMPT,
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "query": query,
                    "intents": [intent.value for intent in expected_intents],
                }
            ),
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
        provided_by_model: dict[IntentType, str] = {}

        # Log extracted intents with parameters
        logger.info(
            "[intent-detection-structured] Detected %d intent(s):",
            len(structured_intents.intents),
        )
        for idx, intent_ctx in enumerate(structured_intents.intents):
            logger.info(
                "[intent-detection-structured]   Intent %d: %s with context='%s'",
                idx + 1,
                intent_ctx.intent.value,
                intent_ctx.trimmed_context(),
            )
            provided_by_model.setdefault(intent_ctx.intent, intent_ctx.trimmed_context())

        # Warn if the model hallucinated extra intents
        unexpected = [
            intent_ctx.intent.value
            for intent_ctx in structured_intents.intents
            if intent_ctx.intent not in expected_intents
        ]
        if unexpected:
            logger.warning(
                "[intent-detection-structured] Ignoring unexpected intent(s) from structured extraction: %s",
                unexpected,
            )

        results: list[IntentWithContext] = []
        for intent in expected_intents:
            context = provided_by_model.get(intent, "")
            if context:
                results.append(IntentWithContext(intent=intent, context_text=context))
                continue

            logger.warning(
                "[intent-detection-structured] No context returned for intent=%s; using full query fallback",
                intent.value,
            )
            results.append(IntentWithContext(intent=intent, context_text=query.strip() or ""))

        return results

    except Exception:  # pylint: disable=broad-except
        logger.error(
            "[intent-detection-structured] Structured detection failed, falling back to simple detection",
            exc_info=True,
        )
        # Fallback: use the full query for every expected intent
        fallback_context = query.strip()
        if not fallback_context:
            logger.warning(
                "[intent-detection-structured] Query is empty after strip; fallback context will be empty"
            )
        return [
            IntentWithContext(intent=intent, context_text=fallback_context)
            for intent in expected_intents
        ]


def generate_continuation_actions(
    client: OpenAI, query: str, intents: list[IntentType], target_language: str = "en"
) -> list[str]:
    """Generate complete continuation questions for each intent using LLM.

    This function takes a multi-intent query and generates complete, grammatically
    correct continuation questions in the target language.

    For example:
    - Query: "Tell me about Barnabas, keywords in Gen 4:2, help translating John 1:4"
    - Intents: [GET_BIBLE_TRANSLATION_ASSISTANCE, GET_PASSAGE_KEYWORDS, GET_TRANSLATION_HELPS]
    - Target Language: "en"
    - Returns: [
        "Would you like me to tell you about Barnabas in the Bible?",
        "Would you like me to show key terms from Gen 4:2?",
        "Would you like me to provide translation helps for John 1:4?"
      ]

    Args:
        client: OpenAI client instance
        query: The user's original query
        intents: List of detected intents in order
        target_language: ISO 639-1 language code for the questions (default: "en")

    Returns:
        List of complete continuation questions, one per intent, in the same order
    """
    logger.info(
        "[continuation-questions] Generating continuation questions for %d intents in language '%s'",
        len(intents),
        target_language,
    )

    intent_names = [intent.value for intent in intents]

    prompt = CONTINUATION_ACTION_GENERATION_PROMPT.format(
        query=query, intents=json.dumps(intent_names), target_language=target_language
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Cheap and fast for this task
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=300,  # Increased for complete questions
            store=False,
        )

        usage = getattr(response, "usage", None)
        track_openai_usage(usage, "gpt-4o-mini", _extract_cached_input_tokens, add_tokens)

        content = response.choices[0].message.content
        if not content:
            logger.warning("[continuation-questions] Empty response from LLM")
            return ["Would you like me to continue with that request?"] * len(intents)

        # Parse JSON array
        questions = json.loads(content.strip())

        # Validate we got the right number
        if not isinstance(questions, list) or len(questions) != len(intents):
            logger.warning(
                "[continuation-questions] Expected %d questions, got %s. Using fallback.",
                len(intents),
                len(questions) if isinstance(questions, list) else "non-list",
            )
            return ["Would you like me to continue with that request?"] * len(intents)

        logger.info("[continuation-questions] Generated questions: %s", questions)
        return questions

    except Exception:  # pylint: disable=broad-except
        logger.error("[continuation-questions] Error generating questions", exc_info=True)
        # Fallback to generic question
        return ["Would you like me to continue with that request?"] * len(intents)


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
    "CONTINUATION_ACTION_GENERATION_PROMPT",
    "DETECT_LANGUAGE_AGENT_SYSTEM_PROMPT",
    "PreprocessorResult",
    "resolve_agentic_strength",
    "model_for_agentic_strength",
    "detect_language",
    "determine_query_language",
    "determine_intents",
    "determine_intents_structured",
    "generate_continuation_actions",
    "is_affirmative_response_to_continuation",
    "preprocess_user_query",
]
INTENT_CLASSIFICATION_STRUCTURED_PROMPT = """
You support BT Servant by extracting context snippets for already-classified user intents.

The user will send a JSON payload with two keys:
- `query`: the user's latest message.
- `intents`: an ordered array of intent strings that have already been approved. You MUST honor this list exactly.

Your task:
- For every intent in the `intents` array, produce a `context_text` string that captures only the portion(s) of `query`
  relevant to that intent.
- Return a JSON object shaped like:
  {
    "intents": [
      {"intent": "<first-intent>", "context_text": "<snippet>"},
      {"intent": "<second-intent>", "context_text": "<snippet>"},
      ...
    ]
  }

Critical rules:
1. Preserve the ordering and spelling of the provided intents. Do not add, remove, rename, or reorder them.
2. Each `context_text` should quote or lightly paraphrase the user's wording so the downstream handler can operate on it directly.
3. If multiple non-contiguous spans apply to the same intent, rewrite them into fluent sentences while staying faithful to the user.
4. If you cannot find intent-specific wording, set `context_text` to an empty string rather than inventing details.
5. Never emit intents that are not in the provided list.
"""
