"""
Parse the Berean Standard Bible (BSB) raw text into per-book JSON files.

Source: https://bereanbible.com/bsb.txt

Output: sources/bible_data/en/bsb/<book>.json
Each file is a JSON array of objects with shape:
  { "reference": "Gen 1:1", "text": "In the beginning..." }

Notes:
- Uses a fixed mapping from full book names in the source text to
  3-letter abbreviations and file basenames (lowercase) for output.
- Reference abbreviations use title-case (e.g., "Gen", "1Sa").
"""
from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
from urllib.request import urlopen


BSB_URL = "https://bereanbible.com/bsb.txt"


@dataclass(frozen=True)
class BookMap:
    """Small mapping entry for one book.

    Attributes
    - file_stem: Lowercase file stem (e.g., "gen", "1sa").
    - ref_abbr: Title-case reference abbreviation (e.g., "Gen", "1Sa").
    """
    file_stem: str
    ref_abbr: str


# Mapping from source book names to our abbreviations and filenames
BOOK_MAP: Dict[str, BookMap] = {
    # Pentateuch
    "Genesis": BookMap("gen", "Gen"),
    "Exodus": BookMap("exo", "Exo"),
    "Leviticus": BookMap("lev", "Lev"),
    "Numbers": BookMap("num", "Num"),
    "Deuteronomy": BookMap("deu", "Deu"),
    # History
    "Joshua": BookMap("jos", "Jos"),
    "Judges": BookMap("jdg", "Jdg"),
    "Ruth": BookMap("rut", "Rut"),
    "1 Samuel": BookMap("1sa", "1Sa"),
    "2 Samuel": BookMap("2sa", "2Sa"),
    "1 Kings": BookMap("1ki", "1Ki"),
    "2 Kings": BookMap("2ki", "2Ki"),
    "1 Chronicles": BookMap("1ch", "1Ch"),
    "2 Chronicles": BookMap("2ch", "2Ch"),
    "Ezra": BookMap("ezr", "Ezr"),
    "Nehemiah": BookMap("neh", "Neh"),
    "Esther": BookMap("est", "Est"),
    # Poetry/Wisdom
    "Job": BookMap("job", "Job"),
    "Psalm": BookMap("psa", "Psa"),  # BSB uses singular per-psalm lines
    "Proverbs": BookMap("pro", "Pro"),
    "Ecclesiastes": BookMap("ecc", "Ecc"),
    "Song of Solomon": BookMap("sos", "Sos"),
    # Major Prophets
    "Isaiah": BookMap("isa", "Isa"),
    "Jeremiah": BookMap("jer", "Jer"),
    "Lamentations": BookMap("lam", "Lam"),
    "Ezekiel": BookMap("eze", "Eze"),
    "Daniel": BookMap("dan", "Dan"),
    # Minor Prophets
    "Hosea": BookMap("hos", "Hos"),
    "Joel": BookMap("joe", "Joe"),
    "Amos": BookMap("amo", "Amo"),
    "Obadiah": BookMap("oba", "Oba"),
    "Jonah": BookMap("jon", "Jon"),
    "Micah": BookMap("mic", "Mic"),
    "Nahum": BookMap("nah", "Nah"),
    "Habakkuk": BookMap("hab", "Hab"),
    "Zephaniah": BookMap("zep", "Zep"),
    "Haggai": BookMap("hag", "Hag"),
    "Zechariah": BookMap("zec", "Zec"),
    "Malachi": BookMap("mal", "Mal"),
    # Gospels/Acts
    "Matthew": BookMap("mat", "Mat"),
    "Mark": BookMap("mar", "Mar"),
    "Luke": BookMap("luk", "Luk"),
    "John": BookMap("joh", "Joh"),
    "Acts": BookMap("act", "Act"),
    # Paul’s Epistles
    "Romans": BookMap("rom", "Rom"),
    "1 Corinthians": BookMap("1co", "1Co"),
    "2 Corinthians": BookMap("2co", "2Co"),
    "Galatians": BookMap("gal", "Gal"),
    "Ephesians": BookMap("eph", "Eph"),
    "Philippians": BookMap("php", "Php"),
    "Colossians": BookMap("col", "Col"),
    "1 Thessalonians": BookMap("1th", "1Th"),
    "2 Thessalonians": BookMap("2th", "2Th"),
    "1 Timothy": BookMap("1ti", "1Ti"),
    "2 Timothy": BookMap("2ti", "2Ti"),
    "Titus": BookMap("tit", "Tit"),
    "Philemon": BookMap("phm", "Phm"),
    # General Epistles + Revelation
    "Hebrews": BookMap("heb", "Heb"),
    "James": BookMap("jas", "Jas"),
    "1 Peter": BookMap("1pe", "1Pe"),
    "2 Peter": BookMap("2pe", "2Pe"),
    "1 John": BookMap("1jo", "1Jo"),
    "2 John": BookMap("2jo", "2Jo"),
    "3 John": BookMap("3jo", "3Jo"),
    "Jude": BookMap("jud", "Jud"),
    "Revelation": BookMap("rev", "Rev"),
}


# Regex to match lines like: "Genesis 1:1\tIn the beginning..."
VERSE_RE = re.compile(
    r"^(?P<book>[1-3] [A-Za-z]+|[A-Za-z]+(?: [A-Za-z]+)*) "
    r"(?P<ch>\d+):(?P<v>\d+)\t(?P<text>.+)$"
)


def fetch_bsb_text(url: str = BSB_URL) -> str:
    """Fetch the raw BSB text from the upstream URL and decode as UTF-8."""
    with urlopen(url) as resp:  # nosec: B310 - trusted upstream text file
        raw = resp.read()
    # Strip a possible UTF-8 BOM if present
    return raw.decode("utf-8-sig")


def parse_lines_to_entries(lines: List[str]) -> Dict[str, List[Tuple[str, str]]]:
    """Parse raw lines into a mapping of book file stems -> list of (reference, text).

    Returns a dict where key is the lowercase file stem (e.g., "gen", "1sa")
    and value is a list of (reference, verse_text) tuples in order.
    """
    grouped: Dict[str, List[Tuple[str, str]]] = {}

    for line in lines:
        line = line.rstrip("\n\r")
        if not line:
            continue
        m = VERSE_RE.match(line)
        if not m:
            # Skip headers or any non-verse lines
            continue

        book_name = m.group("book")
        chap = m.group("ch")
        verse = m.group("v")
        text = m.group("text").strip()

        mapping = BOOK_MAP.get(book_name)
        if mapping is None:
            raise KeyError(f"Unrecognized book name from source: {book_name!r}")

        ref = f"{mapping.ref_abbr} {chap}:{verse}"
        grouped.setdefault(mapping.file_stem, []).append((ref, text))

    return grouped


def write_books_to_json(target_root: Path, data: Dict[str, List[Tuple[str, str]]]) -> None:
    """Write grouped verses to JSON files under the target root."""
    target_root.mkdir(parents=True, exist_ok=True)
    for file_stem, entries in data.items():
        out_path = target_root / f"{file_stem}.json"
        payload = [{"reference": ref, "text": txt} for ref, txt in entries]
        # Ensure ascii is not enforced; keep Unicode punctuation as-is
        out_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


def main(argv: List[str]) -> int:
    """Entry point for one-off parse: fetch, parse, and write per-book JSONs."""
    # Optional first arg: custom output directory (defaults to ./sources/bible_data/en/bsb)
    out_dir = (
        Path(argv[1]) if len(argv) > 1 else Path("sources") / "bible_data" / "en" / "bsb"
    )
    print(f"Fetching BSB text from {BSB_URL}...")
    text = fetch_bsb_text(BSB_URL)
    lines = text.splitlines()
    print("Parsing verses...")
    grouped = parse_lines_to_entries(lines)
    print(f"Writing {len(grouped)} books to {out_dir}...")
    write_books_to_json(out_dir, grouped)
    total = sum(len(v) for v in grouped.values())
    print(f"Done. Wrote {total} verses across {len(grouped)} books.")
    return 0


if __name__ == "__main__":  # pragma: no cover - one-off script
    raise SystemExit(main(sys.argv))
