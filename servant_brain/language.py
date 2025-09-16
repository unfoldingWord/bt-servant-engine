"""Language-related models and prompts used across the brain."""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel


class Language(str, Enum):
    """Supported ISO 639-1 language codes for responses/messages."""

    ENGLISH = "en"
    ARABIC = "ar"
    FRENCH = "fr"
    SPANISH = "es"
    HINDI = "hi"
    RUSSIAN = "ru"
    INDONESIAN = "id"
    SWAHILI = "sw"
    PORTUGUESE = "pt"
    MANDARIN = "zh"
    DUTCH = "nl"
    OTHER = "Other"


class ResponseLanguage(BaseModel):
    """Model for parsing/validating the detected response language."""

    language: Language


class MessageLanguage(BaseModel):
    """Model for parsing/validating the detected language of a message."""

    language: Language


SET_RESPONSE_LANGUAGE_AGENT_SYSTEM_PROMPT = """
Task: Determine the language the user wants responses in, based on conversation context and the latest message.

Allowed outputs: en, ar, fr, es, hi, ru, id, sw, pt, zh, nl, Other

Instructions:
- Use conversation history and the most recent message to infer the user's desired response language.
- Only return one of the allowed outputs. If unclear or unsupported, return Other.
- Consider explicit requests like "reply in French" or language names/codes.
- Output must match the provided schema with no additional prose.
"""


DETECT_LANGUAGE_AGENT_SYSTEM_PROMPT = """
Task: Detect the language of the supplied user text and return the ISO 639-1 code from the allowed set.

Allowed outputs: en, ar, fr, es, hi, ru, id, sw, pt, zh, nl, Other

Bible context: Bible book abbreviations are language-neutral (e.g., Gen, Exo, Lev, Num, Deu, Dan, Joh, Rom, 1Co, 2Co,
Gal, Eph, Php, Col, 1Th, 2Th, 1Ti, 2Ti, Tit, Phm, Heb, Jas, 1Pe, 2Pe, 1Jo, 2Jo, 3Jo, Jud, Rev). The token "Dan"
often denotes the book Daniel and must NOT be interpreted as Indonesian "dan" ("and") when it appears as a book
abbreviation near a chapter/verse reference (e.g., "Dan 1:1"). Treat such abbreviations and references as language-
neutral signal.

Ambiguity rule: If the text is mixed or ambiguous, prefer English (en), especially when common English instruction
keywords are present (e.g., summarize, explain, what, who, why, how).

Output format: Return only structured output matching the schema { "language": <one of the allowed outputs> } with no
additional prose.

Disambiguation examples:
- text: "summarize Dan 1:1" -> { "language": "en" }
- text: "tolong ringkas Dan 1:1" -> { "language": "id" }
- text: "explain Joh 3:16" -> { "language": "en" }
- text: "ringkas Yoh 3:16" -> { "language": "id" }
"""


TARGET_TRANSLATION_LANGUAGE_AGENT_SYSTEM_PROMPT = """
Task: Determine the target language the user is asking the system to translate scripture into, based solely on the
user's latest message. Return an ISO 639-1 code from the allowed set.

Allowed outputs: en, ar, fr, es, hi, ru, id, sw, pt, zh, nl, Other

Rules:
- Identify explicit target-language mentions (language names, codes, or phrases like "into Russian", "to es",
  "in French").
- If no target language is explicitly specified, return Other. Do NOT infer a target from the message's language.
- Output must match the provided schema exactly with no extra prose.

Examples:
- message: "translate John 3:16 into Russian" -> { "language": "ru" }
- message: "please translate Mark 1 in Spanish" -> { "language": "es" }
- message: "translate Matthew 2" -> { "language": "Other" }
"""


__all__ = [
    "Language",
    "ResponseLanguage",
    "MessageLanguage",
    "SET_RESPONSE_LANGUAGE_AGENT_SYSTEM_PROMPT",
    "DETECT_LANGUAGE_AGENT_SYSTEM_PROMPT",
    "TARGET_TRANSLATION_LANGUAGE_AGENT_SYSTEM_PROMPT",
]
