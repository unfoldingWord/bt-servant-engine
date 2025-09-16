"""Shared runtime dependencies for servant brain nodes."""
# pylint: disable=duplicate-code
from __future__ import annotations

from openai import OpenAI

from config import config

open_ai_client = OpenAI(api_key=config.OPENAI_API_KEY)

supported_language_map = {
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

RELEVANCE_CUTOFF = 0.65
TOP_K = 5

__all__ = [
    "open_ai_client",
    "supported_language_map",
    "LANGUAGE_UNKNOWN",
    "RELEVANCE_CUTOFF",
    "TOP_K",
]
