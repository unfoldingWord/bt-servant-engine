"""Response shaping helpers for scripture payloads.

Kept small and pure so handlers can reuse them.
"""

from __future__ import annotations

from typing import Any
import re


def _norm_ws(text: str) -> str:
    """Normalize whitespace to single spaces and trim."""
    return re.sub(r"\s+", " ", str(text)).strip()


def make_scripture_response(
    *,
    content_language: str,
    header_book: str,
    header_suffix: str,
    scripture_text: str,
    header_is_translated: bool,
) -> dict:
    """Build the standard scripture response payload used across intents."""
    return {
        "suppress_translation": True,
        "content_language": content_language,
        "header_is_translated": header_is_translated,
        "segments": [
            {"type": "header_book", "text": header_book},
            {"type": "header_suffix", "text": header_suffix},
            {"type": "scripture", "text": scripture_text},
        ],
    }


def make_scripture_response_from_translated(
    *,
    canonical_book: str,
    header_suffix: str,
    translated: Any,
) -> dict:
    """Adapt a TranslatedPassage-like object to the standard response payload."""
    return make_scripture_response(
        content_language=str(translated.content_language.value),
        header_book=translated.header_book or canonical_book,
        header_suffix=translated.header_suffix or header_suffix,
        scripture_text=_norm_ws(translated.body),
        header_is_translated=True,
    )
