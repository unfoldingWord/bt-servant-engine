"""Helpers to parse USFM and emit per-verse JSON entries.

# pylint: disable=duplicate-code

Provides:
- `USFM_CODE_TO_BOOK`: map from USFM codes (e.g., GEN) to canonical English names.
- `Verse`: typed container for parsed verse rows.
- `parse_usfm_verses(Path) -> list[Verse]`: robust verse-oriented parser using \\c/\\v markers.
- `format_reference`, `build_book_entries`, `book_output_paths` to mirror the BSB JSON format.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import re

from .bsb import BOOK_MAP


USFM_CODE_TO_BOOK: Dict[str, str] = {
    # OT
    "GEN": "Genesis",
    "EXO": "Exodus",
    "LEV": "Leviticus",
    "NUM": "Numbers",
    "DEU": "Deuteronomy",
    "JOS": "Joshua",
    "JDG": "Judges",
    "RUT": "Ruth",
    "1SA": "1 Samuel",
    "2SA": "2 Samuel",
    "1KI": "1 Kings",
    "2KI": "2 Kings",
    "1CH": "1 Chronicles",
    "2CH": "2 Chronicles",
    "EZR": "Ezra",
    "NEH": "Nehemiah",
    "EST": "Esther",
    "JOB": "Job",
    "PSA": "Psalm",
    "PRO": "Proverbs",
    "ECC": "Ecclesiastes",
    "SNG": "Song of Solomon",
    "ISA": "Isaiah",
    "JER": "Jeremiah",
    "LAM": "Lamentations",
    "EZK": "Ezekiel",
    "DAN": "Daniel",
    "HOS": "Hosea",
    "JOL": "Joel",
    "AMO": "Amos",
    "OBA": "Obadiah",
    "JON": "Jonah",
    "MIC": "Micah",
    "NAM": "Nahum",
    "HAB": "Habakkuk",
    "ZEP": "Zephaniah",
    "HAG": "Haggai",
    "ZEC": "Zechariah",
    "MAL": "Malachi",
    # NT
    "MAT": "Matthew",
    "MRK": "Mark",
    "LUK": "Luke",
    "JHN": "John",
    "ACT": "Acts",
    "ROM": "Romans",
    "1CO": "1 Corinthians",
    "2CO": "2 Corinthians",
    "GAL": "Galatians",
    "EPH": "Ephesians",
    "PHP": "Philippians",
    "COL": "Colossians",
    "1TH": "1 Thessalonians",
    "2TH": "2 Thessalonians",
    "1TI": "1 Timothy",
    "2TI": "2 Timothy",
    "TIT": "Titus",
    "PHM": "Philemon",
    "HEB": "Hebrews",
    "JAS": "James",
    "1PE": "1 Peter",
    "2PE": "2 Peter",
    "1JN": "1 John",
    "2JN": "2 John",
    "3JN": "3 John",
    "JUD": "Jude",
    "REV": "Revelation",
}


@dataclass(frozen=True)
class Verse:
    """Parsed verse unit from a USFM file."""

    book: str
    chapter: int
    verse: int
    text: str


def _strip_usfm_inline(text: str) -> str:
    """Normalize verse text: drop inline attributes/markers and collapse whitespace.

    - Replaces "\\w inner|attr... \\w*" with just "inner".
    - Removes stray USFM markers (e.g., \\s5, \\p) while keeping verse content.
    - Collapses internal whitespace to single spaces.
    """
    if not text:
        return ""

    def _strip_w(m: "re.Match[str]") -> str:  # type: ignore[name-defined]
        inner = m.group(1)
        return inner.split("|", 1)[0]

    text = re.sub(r"\\w\s+(.*?)\\w\*", _strip_w, text)
    text = re.sub(r"\\(?!v\b)[A-Za-z0-9]+\*?", "", text)
    text = re.sub(r"\|[^\s]+", "", text)
    return " ".join(text.split())


def parse_usfm_verses(path: Path) -> List[Verse]:
    """Parse USFM lines into a list of Verse objects using \\c/\\v markers.

    Ignores non-verse content and preserves multi-line verse bodies.
    """
    code = path.stem.split("-", 1)[-1].upper() if "-" in path.stem else path.stem.upper()
    book = USFM_CODE_TO_BOOK.get(code, code)

    verses: List[Verse] = []
    cur_ch: int | None = None
    cur_vs: int | None = None
    parts: List[str] = []

    re_ch = re.compile(r"^\\c\s+(\d+)")
    re_vs = re.compile(r"^\\v\s+(\d+)\s+(.*)$")

    def flush() -> None:
        nonlocal cur_ch, cur_vs, parts
        if cur_ch is None or cur_vs is None:
            return
        raw = " ".join(parts).strip()
        verses.append(Verse(book=book, chapter=cur_ch, verse=cur_vs, text=_strip_usfm_inline(raw)))
        cur_vs = None
        parts = []

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        m = re_ch.match(line)
        if m:
            flush()
            cur_ch = int(m.group(1))
            continue
        m = re_vs.match(line)
        if m and cur_ch is not None:
            flush()
            cur_vs = int(m.group(1))
            parts.append(m.group(2))
        elif cur_vs is not None:
            parts.append(line)

    flush()
    return verses


def extract_book_title(path: Path) -> Tuple[str, str | None]:
    """Return (canonical_book, localized_title) by scanning the USFM header (\\h).

    Falls back to None when no \\h line is present. Canonical book is derived
    from the filename USFM code as in parse_usfm_verses.
    """
    code = path.stem.split("-", 1)[-1].upper() if "-" in path.stem else path.stem.upper()
    canonical = USFM_CODE_TO_BOOK.get(code, code)
    title: str | None = None
    try:
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line:
                continue
            if line.startswith("\\h "):
                title = line[3:].strip()
                # strip any trailing markers defensively
                title = re.sub(r"\\[A-Za-z0-9]+\*?", "", title).strip()
                break
    except Exception:  # noqa: BLE001 - tolerate malformed headers in some USFM sources  # pylint: disable=broad-except
        # Be permissive; title is optional
        title = None
    return canonical, title


def format_reference(book: str, chapter: int, verse: int) -> str:
    """Format a BSB-style reference token, e.g., "Gen 1:1"."""
    abbr = BOOK_MAP[book]["ref_abbr"]
    return f"{abbr} {chapter}:{verse}"


def build_book_entries(verses: Iterable[Verse]) -> Dict[str, List[Dict[str, str]]]:
    """Group verses by book and convert to JSON-ready entry dicts."""
    out: Dict[str, List[Dict[str, str]]] = {}
    for v in verses:
        out.setdefault(v.book, []).append(
            {"reference": format_reference(v.book, v.chapter, v.verse), "text": v.text}
        )
    for _book, lst in out.items():
        lst.sort(
            key=lambda e: (
                int(e["reference"].split(" ", 1)[1].split(":", 1)[0]),
                int(e["reference"].split(":", 1)[1]),
            )
        )
    return out


def book_output_paths(root: Path) -> Dict[str, Path]:
    """Return mapping canonical book name -> output JSON path under root."""
    mapping: Dict[str, Path] = {}
    for book, meta in BOOK_MAP.items():
        mapping[book] = root / f"{meta['file_stem']}.json"
    return mapping


def build_dataset(src_dir: Path, out_root: Path) -> None:
    """Build per-book JSON files from a directory of USFM files.

    - Parses all .usfm files under `src_dir`.
    - Groups verses and writes JSON under `out_root` using BSB stems.
    """
    out_root.mkdir(parents=True, exist_ok=True)
    files = sorted(str(p) for p in src_dir.glob("*.usfm"))
    if not files:
        raise FileNotFoundError(f"No USFM files found under {src_dir}")

    all_verses: List[Verse] = []
    titles: Dict[str, str] = {}
    for fp in files:
        p = Path(fp)
        all_verses.extend(parse_usfm_verses(p))
        book, t = extract_book_title(p)
        if t:
            titles[book] = t

    by_book = build_book_entries(all_verses)
    expected = book_output_paths(out_root)
    for book, out_path in expected.items():
        entries = by_book.get(book, [])
        content = json.dumps(entries, ensure_ascii=False, indent=2) + "\n"
        out_path.write_text(content, encoding="utf-8")
    # Write titles mapping if any found
    if titles:
        (out_root / "_book_titles.json").write_text(
            json.dumps(titles, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
