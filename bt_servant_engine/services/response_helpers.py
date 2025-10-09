"""Helper functions for processing and partitioning response items."""

from typing import Iterable, cast


def is_protected_response_item(item: dict) -> bool:
    """Return True if a response item carries scripture to protect from changes."""
    body = cast(dict | str, item.get("response"))
    if isinstance(body, dict):
        if body.get("suppress_translation"):
            return True
        if isinstance(body.get("segments"), list):
            segs = cast(list, body.get("segments"))
            return any(isinstance(seg, dict) and seg.get("type") == "scripture" for seg in segs)
    return False


def partition_response_items(responses: Iterable[dict]) -> tuple[list[dict], list[dict]]:
    """Split responses into scripture-protected and normal sets."""
    protected: list[dict] = []
    normal: list[dict] = []
    for item in responses:
        if is_protected_response_item(item):
            protected.append(item)
        else:
            normal.append(item)
    return protected, normal


def normalize_single_response(item: dict) -> dict | str:
    """Return a representation suitable for translation when no combine is needed."""
    body = cast(dict | str, item.get("response"))
    if isinstance(body, str):
        return body
    return item


def sample_for_language_detection(text: str, sample_chars: int = 100) -> str:
    """Return a short prefix ending at a whitespace boundary for detection."""
    trimmed = text.lstrip()
    if len(trimmed) <= sample_chars:
        return trimmed
    snippet = trimmed[:sample_chars]
    parts = snippet.rsplit(maxsplit=1)
    if len(parts) > 1 and parts[0]:
        return parts[0]
    return snippet


__all__ = [
    "is_protected_response_item",
    "partition_response_items",
    "normalize_single_response",
    "sample_for_language_detection",
]
