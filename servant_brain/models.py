"""Pydantic models shared across brain nodes."""
from __future__ import annotations

from pydantic import BaseModel

from servant_brain.language import Language


class TranslatedPassage(BaseModel):
    """Schema for single-call passage translation output."""

    header_book: str
    header_suffix: str
    body: str
    content_language: Language


__all__ = ["TranslatedPassage"]
