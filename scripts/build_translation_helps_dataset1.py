#!/usr/bin/env python3
"""
Builds Dataset 1 for the get-translation-helps intent.

Inputs:
- BSB verse JSONs in `sources/bible_data/en/bsb/<stem>.json` (in this repo).
- TN TSVs in `<data_root>/<tn_subdir>/tn_*.tsv`.
- ULT/GLT USFM files in `<data_root>/<usfm_subdir>/<nn-CODE>.usfm`.

Output:
- Per-book JSON at `sources/translation_helps<suffix>/<stem>.json`.
  Each file is an array of objects like:
  { "reference", "bsb_verse_text", "ult_verse_text", "notes": [ ... ] }

Usage:
  python scripts/build_translation_helps_dataset1.py \
    --data-root /path/to/bt-servant-engine-data-loaders/datasets

  # For Nepali:
  python scripts/build_translation_helps_dataset1.py \
    --data-root /path/to/bt-servant-engine-data-loaders/datasets \
    --language ne

Notes:
- Only verses with at least one TN note are included.
- USFM parsing uses simple "\\c" and "\\v" markers; refine if needed.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

# Map TN/USFM book codes to BSB file stems under sources/bible_data/en
TN_TO_BSB_STEM: Dict[str, str] = {
    # OT
    "GEN": "gen",
    "EXO": "exo",
    "LEV": "lev",
    "NUM": "num",
    "DEU": "deu",
    "JOS": "jos",
    "JDG": "jdg",
    "RUT": "rut",
    "1SA": "1sa",
    "2SA": "2sa",
    "1KI": "1ki",
    "2KI": "2ki",
    "1CH": "1ch",
    "2CH": "2ch",
    "EZR": "ezr",
    "NEH": "neh",
    "EST": "est",
    "JOB": "job",
    "PSA": "psa",
    "PRO": "pro",
    "ECC": "ecc",
    "SNG": "sos",  # Song of Songs
    "ISA": "isa",
    "JER": "jer",
    "LAM": "lam",
    "EZK": "eze",
    "DAN": "dan",
    "HOS": "hos",
    "JOL": "joe",
    "AMO": "amo",
    "OBA": "oba",
    "JON": "jon",
    "MIC": "mic",
    "NAM": "nah",  # Nahum
    "HAB": "hab",
    "ZEP": "zep",
    "HAG": "hag",
    "ZEC": "zec",
    "MAL": "mal",
    # NT
    "MAT": "mat",
    "MRK": "mar",
    "LUK": "luk",
    "JHN": "joh",
    "ACT": "act",
    "ROM": "rom",
    "1CO": "1co",
    "2CO": "2co",
    "GAL": "gal",
    "EPH": "eph",
    "PHP": "php",
    "COL": "col",
    "1TH": "1th",
    "2TH": "2th",
    "1TI": "1ti",
    "2TI": "2ti",
    "TIT": "tit",
    "PHM": "phm",
    "HEB": "heb",
    "JAS": "jas",
    "1PE": "1pe",
    "2PE": "2pe",
    "1JN": "1jo",
    "2JN": "2jo",
    "3JN": "3jo",
    "JUD": "jud",
    "REV": "rev",
}

# Language-specific configuration for data directories and output paths
LANGUAGE_CONFIG: Dict[str, Dict[str, str]] = {
    "en": {
        "tn_subdir": "uw_translation_notes",
        "usfm_subdir": "ult",
        "out_suffix": "",
    },
    "ne": {
        "tn_subdir": "ne_tn",
        "usfm_subdir": "ne_glt",
        "out_suffix": "_ne",
    },
}


@dataclass(frozen=True)
class VerseKey:
    """Chapter/verse key for verse-indexed maps."""

    chapter: int
    verse: int


def _parse_ult_usfm(usfm_path: Path) -> Dict[VerseKey, str]:
    """Parse a ULT USFM file, returning a mapping of (chapter, verse)->verse text.

    Simplified parser: tracks "\\c <n>" for chapters and "\\v <n> <text>" for verses.
    Accepts verse token variants like "1", "1a", or "1-2" by taking the leading integer.
    """
    chapter = 0
    verses: Dict[VerseKey, str] = {}
    verse_re = re.compile(r"^\\\s*v\s+(\S+)\s+(.*)$")  # \v <token> <text>
    chap_re = re.compile(r"^\\\s*c\s+(\d+)\s*$")  # \c <n>
    with usfm_path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            chap_m = chap_re.match(line)
            if chap_m:
                chapter = int(chap_m.group(1))
                continue
            m = verse_re.match(line)
            if not m or chapter == 0:
                continue
            token, text = m.group(1), m.group(2)
            # Extract leading integer from token
            mnum = re.match(r"^(\d+)", token)
            if not mnum:
                continue
            verse_num = int(mnum.group(1))
            key = VerseKey(chapter, verse_num)
            verses[key] = text.strip()
    return verses


def _load_bsb_book(stem: str, repo_root: Path) -> Tuple[Dict[VerseKey, str], Dict[VerseKey, str]]:
    """Load BSB JSON for a book stem, returning:
    - verse_texts: (chapter, verse)->text
    - verse_labels: (chapter, verse)->reference string (e.g., "1Co 1:1")
    """
    bsb_path = repo_root / "sources" / "bible_data" / "en" / "bsb" / f"{stem}.json"
    with bsb_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    verse_texts: Dict[VerseKey, str] = {}
    verse_labels: Dict[VerseKey, str] = {}
    for item in data:
        ref: str = item["reference"]
        text: str = item["text"]
        # Expect refs like "1Co 1:1" or "Gen 1:1"
        try:
            chap_verse = ref.split()[1]
            chap_str, verse_str = chap_verse.split(":", 1)
            key = VerseKey(int(chap_str), int(verse_str))
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"Unexpected BSB reference format: {ref}") from exc
        verse_texts[key] = text
        verse_labels[key] = ref
    return verse_texts, verse_labels


def _parse_tn_tsv(tsv_path: Path) -> Dict[VerseKey, List[dict]]:
    """Parse a TN TSV and aggregate notes by (chapter, verse)."""
    notes: Dict[VerseKey, List[dict]] = {}
    with tsv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            ref = row.get("Reference", "")
            if not ref or ":" not in ref:
                continue  # skip front matter, intros, etc.
            chap_s, verse_s = ref.split(":", 1)
            if not chap_s.isdigit():
                continue
            # take leading int for verse (e.g., "1", "1-2", "1a")
            mnum = re.match(r"^(\d+)", verse_s)
            if not mnum:
                continue
            key = VerseKey(int(chap_s), int(mnum.group(1)))
            entry = {
                "support_reference": row.get("SupportReference", ""),
                "orig_language_quote": row.get("Quote", ""),
                "note": row.get("Note", ""),
            }
            notes.setdefault(key, []).append(entry)
    return notes


@dataclass(frozen=True)
class BookCtx:
    """Context for building a single book."""

    code: str
    stem: str
    ult_dir: Path
    repo_root: Path
    out_dir: Path


def _build_book(tsv_path: Path, ctx: BookCtx) -> tuple[Path, int, int, int]:
    """Build a single book; return (out_path, items, missing_bsb, missing_ult)."""
    usfm_path = next(iter(ctx.ult_dir.glob(f"*-{ctx.code}.usfm")), None)
    if usfm_path is None:
        raise FileNotFoundError(f"No ULT USFM for {ctx.code}")

    bsb_texts, bsb_labels = _load_bsb_book(ctx.stem, ctx.repo_root)
    ult_texts = _parse_ult_usfm(usfm_path)

    out_items: List[dict] = []
    missing_bsb_count = 0
    missing_ult_count = 0
    for key, notes in _parse_tn_tsv(tsv_path).items():
        if key not in bsb_texts or key not in bsb_labels:
            missing_bsb_count += 1
            continue
        ult = ult_texts.get(key)
        if ult is None:
            missing_ult_count += 1
            ult = ""
        out_items.append(
            {
                "reference": bsb_labels[key],
                "bsb_verse_text": bsb_texts[key],
                "ult_verse_text": ult,
                "notes": notes,
            }
        )

    out_items.sort(
        key=lambda item: (
            int(str(item["reference"]).split()[1].split(":", 1)[0]),
            int(str(item["reference"]).split()[1].split(":", 1)[1]),
        )
    )
    out_path = ctx.out_dir / f"{ctx.stem}.json"
    with out_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(out_items, ensure_ascii=False, indent=2))
        f.write("\n")
    return out_path, len(out_items), missing_bsb_count, missing_ult_count


def build_translation_helps(data_root: Path, repo_root: Path, language: str) -> None:
    """Build translation helps dataset files for all available TN TSVs."""
    lang_cfg = LANGUAGE_CONFIG.get(language)
    if not lang_cfg:
        raise ValueError(f"Unsupported language: {language}. Supported: {list(LANGUAGE_CONFIG.keys())}")

    tn_dir = data_root / lang_cfg["tn_subdir"]
    ult_dir = data_root / lang_cfg["usfm_subdir"]
    out_dir = repo_root / "sources" / f"translation_helps{lang_cfg['out_suffix']}"
    out_dir.mkdir(parents=True, exist_ok=True)

    tsv_paths = sorted(tn_dir.glob("tn_*.tsv"))
    if not tsv_paths:
        raise FileNotFoundError(f"No TN TSVs found in {tn_dir}")

    print(f"[info] Building translation helps for language: {language}")
    print(f"[info] TN source: {tn_dir}")
    print(f"[info] USFM source: {ult_dir}")
    print(f"[info] Output: {out_dir}")

    for tsv_path in tsv_paths:
        code = tsv_path.stem.split("_", 1)[1]  # e.g., 1CO
        stem = TN_TO_BSB_STEM.get(code)
        if not stem:
            print(f"[skip] No BSB stem mapping for code {code}")
            continue

        try:
            out_path, count, miss_bsb, miss_ult = _build_book(
                tsv_path,
                BookCtx(
                    code=code,
                    stem=stem,
                    ult_dir=ult_dir,
                    repo_root=repo_root,
                    out_dir=out_dir,
                ),
            )
        except FileNotFoundError:
            print(f"[warn] No ULT USFM for {code}; skipping")
            continue
        msg = f"[ok] Wrote {out_path} with {count} items"
        if miss_bsb:
            msg += f"; missing BSB for {miss_bsb} TN refs"
        if miss_ult:
            msg += f"; missing ULT for {miss_ult} TN refs"
        print(msg)


def main(argv: Iterable[str] | None = None) -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        required=True,
        type=Path,
        help="Path to cloned bt-servant-engine-data-loaders/datasets directory",
    )
    parser.add_argument(
        "--language",
        default="en",
        choices=list(LANGUAGE_CONFIG.keys()),
        help="Language code for translation helps (default: en)",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    repo_root = Path(__file__).resolve().parents[1]
    build_translation_helps(args.data_root, repo_root, args.language)


if __name__ == "__main__":
    main()
