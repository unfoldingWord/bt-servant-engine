#!/usr/bin/env python3
"""
Augment verse JSON files with translation_challenges from unfoldingWord Translation Notes (TSV).

Phase 1 data prep for get-passage-translation-challenges intent:
- Renames data dir outside this script (sources/verse_data).
- Reads TSVs from the external dataset repo and maps notes to verse entries.

For each verse object in sources/verse_data/<stem>.json, ensures a
`note_data` array exists and appends objects of shape:
  { "issue_type": <SupportReference>, "note": <Note> }

Mapping strategy:
- Determine book from TSV filename (e.g., tn_JHN.tsv -> John).
- Use utils.bsb.BOOK_MAP to locate the correct verse_data JSON by canonical name.
- Parse Reference values like "1:1", "1:1-4", or cross-chapter "1:1-2:3"
  (supports hyphen '-' and en-dash '–'). Skip non-verse refs like "front:intro".
- Apply each note to every verse covered by the reference range.

Usage:
  python scripts/enrich_translation_challenges.py [--tn-dir /path/to/uw_translation_notes]

If --tn-dir is omitted, this script will clone the source repo to a temp folder.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.bsb import BOOK_MAP


DATA_ROOT = Path("sources") / "verse_data"


# USFM-ish book code (from TSV filenames) -> Canonical book name (BOOK_MAP key)
USFM_TO_CANONICAL: Dict[str, str] = {
    # Pentateuch
    "GEN": "Genesis", "EXO": "Exodus", "LEV": "Leviticus", "NUM": "Numbers", "DEU": "Deuteronomy",
    # History
    "JOS": "Joshua", "JDG": "Judges", "RUT": "Ruth", "1SA": "1 Samuel", "2SA": "2 Samuel",
    "1KI": "1 Kings", "2KI": "2 Kings", "1CH": "1 Chronicles", "2CH": "2 Chronicles",
    "EZR": "Ezra", "NEH": "Nehemiah", "EST": "Esther",
    # Poetry/Wisdom
    "JOB": "Job", "PSA": "Psalm", "PRO": "Proverbs", "ECC": "Ecclesiastes", "SNG": "Song of Solomon",
    # Major Prophets
    "ISA": "Isaiah", "JER": "Jeremiah", "LAM": "Lamentations", "EZE": "Ezekiel", "DAN": "Daniel",
    # Minor Prophets
    "HOS": "Hosea", "JOL": "Joel", "AMO": "Amos", "OBA": "Obadiah", "JON": "Jonah", "MIC": "Micah",
    "NAM": "Nahum", "HAB": "Habakkuk", "ZEP": "Zephaniah", "HAG": "Haggai", "ZEC": "Zechariah", "MAL": "Malachi",
    # Gospels/Acts
    "MAT": "Matthew", "MRK": "Mark", "LUK": "Luke", "JHN": "John", "ACT": "Acts",
    # Paul’s Epistles
    "ROM": "Romans", "1CO": "1 Corinthians", "2CO": "2 Corinthians", "GAL": "Galatians",
    "EPH": "Ephesians", "PHP": "Philippians", "COL": "Colossians", "1TH": "1 Thessalonians",
    "2TH": "2 Thessalonians", "1TI": "1 Timothy", "2TI": "2 Timothy", "TIT": "Titus", "PHM": "Philemon",
    # General Epistles + Revelation
    "HEB": "Hebrews", "JAS": "James", "1PE": "1 Peter", "2PE": "2 Peter", "1JN": "1 John",
    "2JN": "2 John", "3JN": "3 John", "JUD": "Jude", "REV": "Revelation",
}


@dataclass(frozen=True)
class RefRange:
    start: Tuple[int, int]
    end: Tuple[int, int]


def _clone_dataset_repo(dest: Path) -> Path:
    repo = "https://github.com/unfoldingWord/bt-servant-engine-data-loaders"
    if dest.exists():
        subprocess.run(["rm", "-rf", str(dest)], check=False)
    subprocess.run(["git", "clone", "--depth=1", repo, str(dest)], check=True)
    return dest / "datasets" / "uw_translation_notes"


def _load_json(path: Path) -> List[dict]:
    import json
    return json.loads(path.read_text(encoding="utf-8"))


def _save_json(path: Path, data: List[dict]) -> None:
    import json
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _build_entry_index(entries: List[dict]) -> Dict[Tuple[int, int], dict]:
    idx: Dict[Tuple[int, int], dict] = {}
    for e in entries:
        ref = e.get("reference", "")
        try:
            _, cv = ref.split(" ", 1)
            ch_s, vs_s = cv.split(":", 1)
            ch = int(ch_s)
            vs = int(vs_s)
        except Exception:
            continue
        idx[(ch, vs)] = e
    return idx


def _parse_reference_range(ref_str: str) -> RefRange | None:
    if ":" not in ref_str:
        return None
    if "intro" in ref_str.lower():
        return None

    s = ref_str.replace("\u2013", "-").replace("\u2014", "-").strip()
    parts = s.split("-")
    try:
        start_ch_s, start_vs_s = parts[0].split(":", 1)
        start_ch = int(start_ch_s)
        start_vs = int(start_vs_s)
        if len(parts) == 1:
            end_ch, end_vs = start_ch, start_vs
        else:
            if ":" in parts[1]:
                end_ch_s, end_vs_s = parts[1].split(":", 1)
                end_ch = int(end_ch_s)
                end_vs = int(end_vs_s)
            else:
                end_ch = start_ch
                end_vs = int(parts[1])
    except Exception:
        return None

    if (end_ch, end_vs) < (start_ch, start_vs):
        start_ch, start_vs, end_ch, end_vs = end_ch, end_vs, start_ch, start_vs
    return RefRange(start=(start_ch, start_vs), end=(end_ch, end_vs))


def _select_keys_in_range(keys: Iterable[Tuple[int, int]], rng: RefRange) -> List[Tuple[int, int]]:
    s_ch, s_vs = rng.start
    e_ch, e_vs = rng.end
    out: List[Tuple[int, int]] = []
    for ch, vs in keys:
        if (ch, vs) < (s_ch, s_vs):
            continue
        if (ch, vs) > (e_ch, e_vs):
            continue
        out.append((ch, vs))
    return out


def enrich_book(tsv_path: Path, book_name: str) -> None:
    mapping = BOOK_MAP.get(book_name)
    if not mapping:
        print(f"[WARN] BOOK_MAP has no entry for canonical name: {book_name}")
        return
    json_path = DATA_ROOT / f"{mapping['file_stem']}.json"
    if not json_path.exists():
        print(f"[WARN] verse_data file not found for {book_name}: {json_path}")
        return

    entries = _load_json(json_path)
    idx = _build_entry_index(entries)

    # Reset `note_data` and drop old keys if present
    for e in entries:
        if "translation_challanges" in e:
            e.pop("translation_challanges", None)
        if "translation_challenges" in e:
            e.pop("translation_challenges", None)
        e["note_data"] = []

    with tsv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            ref = (row.get("Reference") or "").strip()
            issue_type = (row.get("SupportReference") or "").strip()
            note = (row.get("Note") or "").strip()

            rng = _parse_reference_range(ref)
            if rng is None:
                continue

            keys = _select_keys_in_range(idx.keys(), rng)
            if not keys:
                continue
            for key in keys:
                e = idx.get(key)
                if e is None:
                    continue
                e.setdefault("note_data", [])
                e["note_data"].append({
                    "issue_type": issue_type,
                    "note": note,
                })

    _save_json(json_path, entries)
    print(f"[OK] Enriched {book_name} -> {json_path} (note_data)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tn-dir", type=str, default="",
                        help="Path to uw_translation_notes directory containing tn_*.tsv files.")
    args = parser.parse_args()

    tn_dir: Path
    if args.tn_dir:
        tn_dir = Path(args.tn_dir)
    else:
        tn_dir = _clone_dataset_repo(Path("/tmp/bt-servant-engine-data-loaders"))

    if not DATA_ROOT.exists():
        raise SystemExit(f"Data root not found: {DATA_ROOT}")

    tsv_paths = sorted(p for p in tn_dir.glob("tn_*.tsv") if p.is_file())
    for p in tsv_paths:
        code = p.stem.replace("tn_", "").upper()
        book = USFM_TO_CANONICAL.get(code)
        if not book:
            print(f"[SKIP] No mapping for {code} ({p.name})")
            continue
        if book not in BOOK_MAP:
            print(f"[SKIP] Canonical book not in BOOK_MAP: {book}")
            continue
        enrich_book(p, book)


if __name__ == "__main__":
    main()
