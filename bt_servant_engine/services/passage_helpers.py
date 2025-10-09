"""Helper functions for Bible passage and book detection."""

import re


def book_patterns(book_map: dict) -> list[tuple[str, str]]:
    """Return (canonical, regex) patterns to detect book mentions (ordered)."""
    pats: list[tuple[str, str]] = []
    for canonical, meta in book_map.items():
        # canonical name as a whole word/phrase
        cn = re.escape(canonical)
        pats.append((canonical, rf"\b{cn}\b"))
        # short ref abbreviation (e.g., Gen, Exo, 1Sa)
        abbr = re.escape(meta.get("ref_abbr", ""))
        if abbr:
            pats.append((canonical, rf"\b{abbr}\b"))
    return pats


def detect_mentioned_books(text: str, book_map: dict) -> list[str]:
    """Detect canonical books mentioned in text, preserving order of appearance."""
    found: list[tuple[int, str]] = []
    lower = text
    for canonical, pattern in book_patterns(book_map):
        for m in re.finditer(pattern, lower, flags=re.IGNORECASE):
            found.append((m.start(), canonical))
    # sort by appearance and dedupe preserving order
    found.sort(key=lambda t: t[0])
    seen = set()
    ordered: list[str] = []
    for _, can in found:
        if can not in seen:
            seen.add(can)
            ordered.append(can)
    return ordered


def choose_primary_book(text: str, candidates: list[str], book_map: dict) -> str | None:
    """Heuristic to pick a primary book when multiple are mentioned.

    Prefer the first mentioned that appears near chapter/verse digits; else None.
    """
    if not candidates:
        return None
    # Build spans for each candidate occurrence
    spans: list[tuple[int, int, str]] = []
    for can, pat in book_patterns(book_map):
        if can not in candidates:
            continue
        for m in re.finditer(pat, text, flags=re.IGNORECASE):
            spans.append((m.start(), m.end(), can))
    spans.sort(key=lambda t: t[0])
    for s_idx, end, can in spans:
        _ = s_idx  # avoid shadowing outer start() function
        window = text[end : end + 12]
        if re.search(r"\d", window):
            return can
    return None


__all__ = ["book_patterns", "detect_mentioned_books", "choose_primary_book"]
