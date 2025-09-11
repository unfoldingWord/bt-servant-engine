"""Minimal Bible locale utilities.

Provides localized book-name lookup with graceful fallback to the canonical
English name when a mapping is not available.
"""
from __future__ import annotations

from typing import Dict


# Canonical (English) names are the keys used throughout utils.bsb BOOK_MAP
_BOOK_LOCALIZATION: Dict[str, Dict[str, str]] = {
    # English canonical (identity mapping by default; shown for clarity)
    "en": {},
    # Example starter for Portuguese (can be expanded over time)
    "pt": {
        # "John": "João",  # enable as mappings are confirmed
        # "Genesis": "Gênesis",
    },
    # Indonesian (partial mapping; expands over time to avoid LLM header translations)
    "id": {
        "Genesis": "Kejadian",
        "Exodus": "Keluaran",
        "Leviticus": "Imamat",
        "Numbers": "Bilangan",
        "Deuteronomy": "Ulangan",
    },
}


def get_book_name(lang: str, canonical_book: str) -> str:
    """Return a localized book name if known; otherwise the canonical name."""
    lang = (lang or "en").lower()
    mapping = _BOOK_LOCALIZATION.get(lang, {})
    return mapping.get(canonical_book, canonical_book)
