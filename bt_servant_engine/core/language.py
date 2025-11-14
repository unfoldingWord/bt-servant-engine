"""Language models and helpers for the BT Servant application."""

from enum import Enum
import re
from typing import Optional, Union

from pydantic import BaseModel, field_validator

# Mapping of ISO 639-1 language codes to friendly names.
# This is intentionally non-exhaustive and serves as a set of display overrides
# for the languages we reference most frequently in user-facing copy.
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
LANGUAGE_OTHER = "other"
_LANGUAGE_CODE_PATTERN = re.compile(r"^[a-z]{2}(?:-[a-z]{2})?$")
_LANGUAGE_NAME_LOOKUP = {name.lower(): code for code, name in SUPPORTED_LANGUAGE_MAP.items()}


class Language(str, Enum):
    """Historical enum for legacy references (kept for compatibility)."""

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


def _normalize_candidate(value: Union[str, "Language", None]) -> Optional[str]:
    normalized: Optional[str] = None
    if value is None:
        normalized = None
    elif isinstance(value, Language):
        normalized = LANGUAGE_OTHER if value is Language.OTHER else value.value
    else:
        candidate = str(value).strip().lower()
        if candidate:
            if candidate == LANGUAGE_OTHER:
                normalized = LANGUAGE_OTHER
            elif _LANGUAGE_CODE_PATTERN.match(candidate):
                normalized = candidate
    return normalized


def normalize_language_code(value: Union[str, "Language", None]) -> Optional[str]:
    """Normalize input into a lowercase ISO 639-1 (optionally xx-yy) code."""
    normalized = _normalize_candidate(value)
    if normalized == LANGUAGE_OTHER:
        return LANGUAGE_OTHER
    return normalized


def normalized_or_other(value: Union[str, "Language", None]) -> str:
    """Normalize to an ISO code; fall back to 'other' when unknown."""
    normalized = normalize_language_code(value)
    return normalized or LANGUAGE_OTHER


def friendly_language_name(
    code: Union[str, "Language", None], *, fallback: str = "that language"
) -> str:
    """Return a printable name for the given language code."""
    normalized = normalize_language_code(code)
    if not normalized or normalized == LANGUAGE_OTHER:
        return fallback
    return SUPPORTED_LANGUAGE_MAP.get(normalized, normalized.title())


def lookup_language_code(name: Optional[str]) -> Optional[str]:
    """Return a best-effort ISO code for a human-readable language name."""
    if not name:
        return None
    normalized = name.strip().lower()
    return _LANGUAGE_NAME_LOOKUP.get(normalized)


class ResponseLanguage(BaseModel):
    """Model for parsing/validating the detected response language."""

    language: str

    @field_validator("language", mode="before")
    @classmethod
    def _coerce_language(cls, value: Union[str, "Language"]) -> str:
        normalized = normalized_or_other(value)
        if normalized == LANGUAGE_OTHER:
            return LANGUAGE_OTHER
        if not normalized:
            raise ValueError("language must be an ISO 639-1 code or 'Other'")
        return normalized


class MessageLanguage(BaseModel):
    """Model for parsing/validating the detected language of a message."""

    language: str

    @field_validator("language", mode="before")
    @classmethod
    def _coerce_language(cls, value: Union[str, "Language"]) -> str:
        normalized = normalized_or_other(value)
        if not normalized or normalized == LANGUAGE_OTHER:
            raise ValueError("message language must be an ISO 639-1 code")
        return normalized


class TranslatedPassage(BaseModel):
    """Schema for single-call passage translation output.

    - header_book: translated book name (e.g., "Иоанн").
    - header_suffix: exact suffix copied from input (e.g., "1:1–7").
    - body: translated passage body with original newlines preserved.
    - content_language: ISO 639-1 code (should equal requested target language).
    """

    header_book: str
    header_suffix: str
    body: str
    content_language: str

    @field_validator("content_language", mode="before")
    @classmethod
    def _coerce_content_language(cls, value: Union[str, "Language"]) -> str:
        normalized = normalize_language_code(value)
        if not normalized or normalized == LANGUAGE_OTHER:
            raise ValueError("content_language must be an ISO 639-1 code")
        return normalized


__all__ = [
    "SUPPORTED_LANGUAGE_MAP",
    "LANGUAGE_UNKNOWN",
    "LANGUAGE_OTHER",
    "Language",
    "ResponseLanguage",
    "MessageLanguage",
    "TranslatedPassage",
    "normalize_language_code",
    "normalized_or_other",
    "friendly_language_name",
    "lookup_language_code",
]
