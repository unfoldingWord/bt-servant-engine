"""Language models and constants for the BT Servant application."""

from enum import Enum

from pydantic import BaseModel

# Mapping of ISO 639-1 language codes to friendly names
SUPPORTED_LANGUAGE_MAP = {
    "en": "English",
    "ar": "Arabic",
    "fr": "French",
    "es": "Spanish",
    "hi": "Hindi",
    "ru": "Russian",
    "id": "Indonesian",
    "sw": "Swahili",
    "pt": "Portuguese",
    "zh": "Mandarin",
    "nl": "Dutch",
}

LANGUAGE_UNKNOWN = "UNKNOWN"


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


class TranslatedPassage(BaseModel):
    """Schema for single-call passage translation output.

    - header_book: translated book name (e.g., "Иоанн").
    - header_suffix: exact suffix copied from input (e.g., "1:1–7").
    - body: translated passage body with original newlines preserved.
    - content_language: ISO 639-1 code (should equal requested target language).
    - follow_up_question: contextual follow-up suggesting next passage or related book.
    """

    header_book: str
    header_suffix: str
    body: str
    content_language: Language
    follow_up_question: str


class RetrievedPassage(BaseModel):
    """Schema for scripture retrieval output.

    - header_book: book name (localized if available).
    - header_suffix: exact suffix for the reference (e.g., "1:1–7").
    - body: passage body with verse text.
    - content_language: ISO 639-1 code of the retrieved scripture.
    - follow_up_question: contextual follow-up suggesting next passage or related book.
    """

    header_book: str
    header_suffix: str
    body: str
    content_language: Language
    follow_up_question: str


__all__ = [
    "SUPPORTED_LANGUAGE_MAP",
    "LANGUAGE_UNKNOWN",
    "Language",
    "ResponseLanguage",
    "MessageLanguage",
    "TranslatedPassage",
    "RetrievedPassage",
]
