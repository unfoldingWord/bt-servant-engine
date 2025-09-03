"""Keyword utilities for per-book topical word selection.

Loads keyword JSONs and filters matches across verse ranges.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

# Map canonical book names to the 3-letter (or 4 with leading digit) codes
# used by the keyword dataset filenames, e.g., sources/keyword_data/keywords_JHN.json
BOOK_TO_CODE: Dict[str, str] = {
    # Pentateuch
    "Genesis": "GEN",
    "Exodus": "EXO",
    "Leviticus": "LEV",
    "Numbers": "NUM",
    "Deuteronomy": "DEU",
    # History
    "Joshua": "JOS",
    "Judges": "JDG",
    "Ruth": "RUT",
    "1 Samuel": "1SA",
    "2 Samuel": "2SA",
    "1 Kings": "1KI",
    "2 Kings": "2KI",
    "1 Chronicles": "1CH",
    "2 Chronicles": "2CH",
    "Ezra": "EZR",
    "Nehemiah": "NEH",
    "Esther": "EST",
    # Poetry/Wisdom
    "Job": "JOB",
    "Psalm": "PSA",
    "Proverbs": "PRO",
    "Ecclesiastes": "ECC",
    "Song of Solomon": "SNG",
    # Major Prophets
    "Isaiah": "ISA",
    "Jeremiah": "JER",
    "Lamentations": "LAM",
    "Ezekiel": "EZK",
    "Daniel": "DAN",
    # Minor Prophets
    "Hosea": "HOS",
    "Joel": "JOL",
    "Amos": "AMO",
    "Obadiah": "OBA",
    "Jonah": "JON",
    "Micah": "MIC",
    "Nahum": "NAM",
    "Habakkuk": "HAB",
    "Zephaniah": "ZEP",
    "Haggai": "HAG",
    "Zechariah": "ZEC",
    "Malachi": "MAL",
    # Gospels/Acts
    "Matthew": "MAT",
    "Mark": "MRK",
    "Luke": "LUK",
    "John": "JHN",
    "Acts": "ACT",
    # Paul’s Epistles
    "Romans": "ROM",
    "1 Corinthians": "1CO",
    "2 Corinthians": "2CO",
    "Galatians": "GAL",
    "Ephesians": "EPH",
    "Philippians": "PHP",
    "Colossians": "COL",
    "1 Thessalonians": "1TH",
    "2 Thessalonians": "2TH",
    "1 Timothy": "1TI",
    "2 Timothy": "2TI",
    "Titus": "TIT",
    "Philemon": "PHM",
    # General Epistles + Revelation
    "Hebrews": "HEB",
    "James": "JAS",
    "1 Peter": "1PE",
    "2 Peter": "2PE",
    "1 John": "1JN",
    "2 John": "2JN",
    "3 John": "3JN",
    "Jude": "JUD",
    "Revelation": "REV",
}


@lru_cache(maxsize=128)
def load_keywords_json(data_root: Path, code: str) -> Dict[str, List[Dict[str, str]]]:
    """Load a keyword JSON for a given book code (cached)."""
    path = data_root / f"keywords_{code}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _in_range(
    ch: int,
    vs: int,
    sel: tuple[int, int | None, int | None, int | None],
) -> bool:
    """Return True if verse (ch, vs) falls within the selection tuple."""
    start_ch, start_vs, end_ch, end_vs = sel
    s_vs = start_vs if start_vs is not None else 1
    e_ch = end_ch if end_ch is not None else start_ch
    e_vs = end_vs if end_vs is not None else 10_000
    if ch < start_ch or ch > e_ch:
        return False
    if ch == start_ch and vs < s_vs:
        return False
    if ch == e_ch and vs > e_vs:
        return False
    return True


def select_keywords(
    data_root: Path,
    canonical_book: str,
    ranges: Iterable[Tuple[int, int | None, int | None, int | None]],
) -> List[str]:
    """Select distinct tw_match keywords for a book across the given ranges.

    ranges: iterable of (start_ch, start_vs, end_ch, end_vs). Use None to denote
            whole chapter/book semantics (mirrors utils.bsb.select_verses input).
    """
    code = BOOK_TO_CODE.get(canonical_book)
    if not code:
        return []
    data = load_keywords_json(data_root, code)

    found: Set[str] = set()
    for sel in ranges:
        for key, entries in data.items():
            try:
                ch_s, vs_s = key.split(":", 1)
                ch_i, vs_i = int(ch_s), int(vs_s)
            except ValueError:
                continue
            if _in_range(ch_i, vs_i, sel):
                for e in entries:
                    tw = e.get("tw_match")
                    if tw:
                        found.add(tw)

    # Case-insensitive sort so capitalized words do not group separately.
    # Use a lambda to satisfy type-checkers/IDEs expecting (str) -> Any.
    return sorted(found, key=lambda s: s.casefold())
