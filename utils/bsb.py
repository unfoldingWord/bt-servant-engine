"""BSB utilities: loading, selecting, and labeling Bible passages.

Provides helpers to load per-book JSON, select verse ranges, and
produce canonical labels for output headers.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


# Centralized mapping of canonical book names to file stem and reference abbreviation.
# File stems correspond to per-book JSONs under a chosen data root
# (e.g., sources/bible_data/en/bsb/<stem>.json). The ref_abbr matches the
# "reference" prefix in those files.
BOOK_MAP: Dict[str, Dict[str, str]] = {
    # Pentateuch
    "Genesis": {"file_stem": "gen", "ref_abbr": "Gen"},
    "Exodus": {"file_stem": "exo", "ref_abbr": "Exo"},
    "Leviticus": {"file_stem": "lev", "ref_abbr": "Lev"},
    "Numbers": {"file_stem": "num", "ref_abbr": "Num"},
    "Deuteronomy": {"file_stem": "deu", "ref_abbr": "Deu"},
    # History
    "Joshua": {"file_stem": "jos", "ref_abbr": "Jos"},
    "Judges": {"file_stem": "jdg", "ref_abbr": "Jdg"},
    "Ruth": {"file_stem": "rut", "ref_abbr": "Rut"},
    "1 Samuel": {"file_stem": "1sa", "ref_abbr": "1Sa"},
    "2 Samuel": {"file_stem": "2sa", "ref_abbr": "2Sa"},
    "1 Kings": {"file_stem": "1ki", "ref_abbr": "1Ki"},
    "2 Kings": {"file_stem": "2ki", "ref_abbr": "2Ki"},
    "1 Chronicles": {"file_stem": "1ch", "ref_abbr": "1Ch"},
    "2 Chronicles": {"file_stem": "2ch", "ref_abbr": "2Ch"},
    "Ezra": {"file_stem": "ezr", "ref_abbr": "Ezr"},
    "Nehemiah": {"file_stem": "neh", "ref_abbr": "Neh"},
    "Esther": {"file_stem": "est", "ref_abbr": "Est"},
    # Poetry/Wisdom
    "Job": {"file_stem": "job", "ref_abbr": "Job"},
    "Psalm": {"file_stem": "psa", "ref_abbr": "Psa"},
    "Proverbs": {"file_stem": "pro", "ref_abbr": "Pro"},
    "Ecclesiastes": {"file_stem": "ecc", "ref_abbr": "Ecc"},
    "Song of Solomon": {"file_stem": "sos", "ref_abbr": "Sos"},
    # Major Prophets
    "Isaiah": {"file_stem": "isa", "ref_abbr": "Isa"},
    "Jeremiah": {"file_stem": "jer", "ref_abbr": "Jer"},
    "Lamentations": {"file_stem": "lam", "ref_abbr": "Lam"},
    "Ezekiel": {"file_stem": "eze", "ref_abbr": "Eze"},
    "Daniel": {"file_stem": "dan", "ref_abbr": "Dan"},
    # Minor Prophets
    "Hosea": {"file_stem": "hos", "ref_abbr": "Hos"},
    "Joel": {"file_stem": "joe", "ref_abbr": "Joe"},
    "Amos": {"file_stem": "amo", "ref_abbr": "Amo"},
    "Obadiah": {"file_stem": "oba", "ref_abbr": "Oba"},
    "Jonah": {"file_stem": "jon", "ref_abbr": "Jon"},
    "Micah": {"file_stem": "mic", "ref_abbr": "Mic"},
    "Nahum": {"file_stem": "nah", "ref_abbr": "Nah"},
    "Habakkuk": {"file_stem": "hab", "ref_abbr": "Hab"},
    "Zephaniah": {"file_stem": "zep", "ref_abbr": "Zep"},
    "Haggai": {"file_stem": "hag", "ref_abbr": "Hag"},
    "Zechariah": {"file_stem": "zec", "ref_abbr": "Zec"},
    "Malachi": {"file_stem": "mal", "ref_abbr": "Mal"},
    # Gospels/Acts
    "Matthew": {"file_stem": "mat", "ref_abbr": "Mat"},
    "Mark": {"file_stem": "mar", "ref_abbr": "Mar"},
    "Luke": {"file_stem": "luk", "ref_abbr": "Luk"},
    "John": {"file_stem": "joh", "ref_abbr": "Joh"},
    "Acts": {"file_stem": "act", "ref_abbr": "Act"},
    # Paul’s Epistles
    "Romans": {"file_stem": "rom", "ref_abbr": "Rom"},
    "1 Corinthians": {"file_stem": "1co", "ref_abbr": "1Co"},
    "2 Corinthians": {"file_stem": "2co", "ref_abbr": "2Co"},
    "Galatians": {"file_stem": "gal", "ref_abbr": "Gal"},
    "Ephesians": {"file_stem": "eph", "ref_abbr": "Eph"},
    "Philippians": {"file_stem": "php", "ref_abbr": "Php"},
    "Colossians": {"file_stem": "col", "ref_abbr": "Col"},
    "1 Thessalonians": {"file_stem": "1th", "ref_abbr": "1Th"},
    "2 Thessalonians": {"file_stem": "2th", "ref_abbr": "2Th"},
    "1 Timothy": {"file_stem": "1ti", "ref_abbr": "1Ti"},
    "2 Timothy": {"file_stem": "2ti", "ref_abbr": "2Ti"},
    "Titus": {"file_stem": "tit", "ref_abbr": "Tit"},
    "Philemon": {"file_stem": "phm", "ref_abbr": "Phm"},
    # General Epistles + Revelation
    "Hebrews": {"file_stem": "heb", "ref_abbr": "Heb"},
    "James": {"file_stem": "jas", "ref_abbr": "Jas"},
    "1 Peter": {"file_stem": "1pe", "ref_abbr": "1Pe"},
    "2 Peter": {"file_stem": "2pe", "ref_abbr": "2Pe"},
    "1 John": {"file_stem": "1jo", "ref_abbr": "1Jo"},
    "2 John": {"file_stem": "2jo", "ref_abbr": "2Jo"},
    "3 John": {"file_stem": "3jo", "ref_abbr": "3Jo"},
    "Jude": {"file_stem": "jud", "ref_abbr": "Jud"},
    "Revelation": {"file_stem": "rev", "ref_abbr": "Rev"},
}


# Common alias/abbreviation normalization to canonical names used in BOOK_MAP
BOOK_ALIASES: Dict[str, str] = {
    # Gospels
    "jn": "John", "jhn": "John", "john": "John",
    "mt": "Matthew", "matt": "Matthew", "matthew": "Matthew",
    "mk": "Mark", "mrk": "Mark", "mark": "Mark",
    "lk": "Luke", "luk": "Luke", "luke": "Luke",
    # Psalms/Song
    "ps": "Psalm", "psa": "Psalm", "psalm": "Psalm", "psalms": "Psalm",
    "sos": "Song of Solomon", "song": "Song of Solomon", "song of songs": "Song of Solomon",
    # 1/2/3 books
    "1jn": "1 John", "1 john": "1 John", "i john": "1 John",
    "2jn": "2 John", "2 john": "2 John", "ii john": "2 John",
    "3jn": "3 John", "3 john": "3 John", "iii john": "3 John",
    "1pet": "1 Peter", "1pe": "1 Peter", "1 peter": "1 Peter", "i peter": "1 Peter",
    "2pet": "2 Peter", "2pe": "2 Peter", "2 peter": "2 Peter", "ii peter": "2 Peter",
    "1sam": "1 Samuel", "1sa": "1 Samuel", "1 samuel": "1 Samuel",
    "2sam": "2 Samuel", "2sa": "2 Samuel", "2 samuel": "2 Samuel",
    "1ki": "1 Kings", "1 kings": "1 Kings", "i kings": "1 Kings",
    "2ki": "2 Kings", "2 kings": "2 Kings", "ii kings": "2 Kings",
    "1ch": "1 Chronicles", "1 chron": "1 Chronicles", "1 chronicles": "1 Chronicles",
    "2ch": "2 Chronicles", "2 chron": "2 Chronicles", "2 chronicles": "2 Chronicles",
    "1th": "1 Thessalonians", "1 thes": "1 Thessalonians", "1 thessalonians": "1 Thessalonians",
    "2th": "2 Thessalonians", "2 thes": "2 Thessalonians", "2 thessalonians": "2 Thessalonians",
    "1ti": "1 Timothy", "1 tim": "1 Timothy", "1 timothy": "1 Timothy",
    "2ti": "2 Timothy", "2 tim": "2 Timothy", "2 timothy": "2 Timothy",
}


def normalize_book_name(name: str) -> str | None:
    """Normalize various aliases/abbreviations to canonical book names.

    Returns None if the name cannot be normalized to a canonical key.
    """
    key = name.strip().lower()
    if key in BOOK_ALIASES:
        return BOOK_ALIASES[key]
    # try title-case direct match
    title = name.strip().title()
    return title if title in BOOK_MAP else None


@lru_cache(maxsize=128)
def load_book_json(data_root: Path, file_stem: str) -> List[Dict[str, str]]:
    """Load a single book JSON file from sources/bible_data/en (cached)."""
    path = data_root / f"{file_stem}.json"
    # Intentionally no logging dependency here to keep utils lightweight
    return json.loads(path.read_text(encoding="utf-8"))


def build_index(entries: List[Dict[str, str]]) -> Dict[Tuple[int, int], Tuple[str, str]]:
    """Build a quick (chapter, verse) -> (ref, text) index from entries."""
    idx: Dict[Tuple[int, int], Tuple[str, str]] = {}
    for e in entries:
        ref = e["reference"]  # e.g., "Joh 3:16"
        parsed = parse_ch_verse_from_reference(ref)
        if parsed is None:
            continue
        ch, vs = parsed
        idx[(ch, vs)] = (ref, e["text"])
    return idx


def parse_ch_verse_from_reference(ref: str) -> Tuple[int, int] | None:
    """Parse a reference token like "Joh 3:16" to (chapter, verse).

    Returns None if parsing fails.
    """
    try:
        _, cv = ref.split(" ", 1)
        ch_s, vs_s = cv.split(":", 1)
        return int(ch_s), int(vs_s)
    except (ValueError, KeyError):
        return None


def select_range(
    idx: Dict[Tuple[int, int], Tuple[str, str]],
    start_ch: int,
    start_vs: int | None,
    end_ch: int | None,
    end_vs: int | None,
) -> List[Tuple[str, str]]:
    """Select verses from index in inclusive range (supports cross-chapter within a book)."""
    items = []
    # Determine bounds; if no verses, select whole chapter(s)
    s_ch = start_ch
    s_vs = start_vs if start_vs is not None else 1
    e_ch = end_ch if end_ch is not None else s_ch
    e_vs = end_vs if end_vs is not None else 10_000  # big sentinel

    for (ch, vs), pair in sorted(idx.items()):
        if (ch < s_ch) or (ch == s_ch and vs < s_vs):
            continue
        if (ch > e_ch) or (ch == e_ch and vs > e_vs):
            break
        items.append(pair)
    return items


def select_verses(
    data_root: Path,
    canonical_book: str,
    ranges: Iterable[Tuple[int, int | None, int | None, int | None]],
) -> List[Tuple[str, str]]:
    """Select verses for a canonical book given a set of ranges.

    ranges: iterable of (start_ch, start_vs, end_ch, end_vs).
    Use None to denote whole chapter/book semantics.
    """
    mapping = BOOK_MAP[canonical_book]
    entries = load_book_json(data_root, mapping["file_stem"])  # cached
    idx = build_index(entries)
    result: List[Tuple[str, str]] = []
    for sc, sv, ec, ev in ranges:
        result.extend(select_range(idx, sc, sv, ec, ev))
    return result


def label_ranges(
    canonical_book: str,
    ranges: List[Tuple[int, int | None, int | None, int | None]],
) -> str:
    """Return a canonical human-readable reference label for the selection.

    Rules:
    - Whole chapter: "Book 3"
    - Same-chapter verse range: "Book 3:1-8" (or single: "Book 3:1")
    - Multi-chapter without verses: "Book 1-4" (no colon)
    - Cross-chapter with verses: "Book 3:16-4:2"
    """
    # Special-case: treat a sentinel full-book range as just the book name.
    # Upstream callers represent a whole-book selection as (1, None, 10_000, None).
    if len(ranges) == 1:
        sc, sv, ec, ev = ranges[0]
        full_book = (
            sc == 1 and sv is None and ev is None and ec is not None and ec >= 10_000
        )
        if full_book:
            return f"{canonical_book}"
    parts: List[str] = []
    for sc, sv, ec, ev in ranges:
        # Whole-chapter(s) selection (no verses specified)
        if sv is None and ev is None:
            if ec is None or ec == sc:
                parts.append(f"{sc}")
            else:
                parts.append(f"{sc}-{ec}")
            continue

        # Same chapter
        if ec is None or ec == sc:
            if sv is None and ev is not None:
                parts.append(f"{sc}:{ev}")
            elif sv is not None and (ev is None or ev == sv):
                parts.append(f"{sc}:{sv}")
            else:
                # both present and differ
                svv = sv if sv is not None else 1
                parts.append(f"{sc}:{svv}-{ev}")
            continue

        # Cross-chapter
        left = f"{sc}:{sv if sv is not None else 1}"
        if ev is None:
            right = f"{ec}"
        else:
            right = f"{ec}:{ev}"
        parts.append(f"{left}-{right}")

    return f"{canonical_book} " + "; ".join(parts)


def clamp_ranges_by_verse_limit(
    data_root: Path,
    canonical_book: str,
    ranges: List[Tuple[int, int | None, int | None, int | None]],
    max_verses: int,
) -> List[Tuple[int, int | None, int | None, int | None]]:
    """Return new ranges clamped to the first `max_verses` verses in reading order.

    Strategy:
    - Expand the input ranges into an ordered list of (chapter, verse) pairs using
      the book's available verse index.
    - Take the first `max_verses` pairs.
    - Compress into chapter-local contiguous verse spans to form ranges of the form
      (ch, start_vs, ch, end_vs). We do not attempt to create cross-chapter spans.
    """
    mapping = BOOK_MAP[canonical_book]
    entries = load_book_json(data_root, mapping["file_stem"])  # cached
    idx = build_index(entries)

    def _in_ranges(ch: int, vs: int) -> bool:
        for sc, sv, ec, ev in ranges:
            s_ch = sc
            s_vs = sv if sv is not None else 1
            e_ch = ec if ec is not None else s_ch
            e_vs = ev if ev is not None else 10_000
            if (ch < s_ch) or (ch == s_ch and vs < s_vs):
                continue
            if (ch > e_ch) or (ch == e_ch and vs > e_vs):
                continue
            return True
        return False

    coords: List[Tuple[int, int]] = []
    for (ch, vs) in sorted(idx.keys()):
        if _in_ranges(ch, vs):
            coords.append((ch, vs))
            if len(coords) >= max_verses:
                break

    if not coords:
        return []

    out: List[Tuple[int, int | None, int | None, int | None]] = []
    cur_ch, start_vs = coords[0]
    prev_vs = start_vs
    for (ch, vs) in coords[1:]:
        if ch == cur_ch and vs == prev_vs + 1:
            prev_vs = vs
            continue
        out.append((cur_ch, start_vs, cur_ch, prev_vs))
        cur_ch, start_vs, prev_vs = ch, vs, vs
    out.append((cur_ch, start_vs, cur_ch, prev_vs))
    return out
