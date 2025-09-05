"""Translation Helps utilities: loading and selecting help entries.

Selects per-verse translation helps from `sources/translation_helps` for a
given canonical book and verse ranges, mirroring the patterns used in
`utils/bsb.py`.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Any

from .bsb import BOOK_MAP, parse_ch_verse_from_reference


@lru_cache(maxsize=128)
def load_book_json(data_root: Path, file_stem: str) -> List[Dict[str, Any]]:
    """Load a single book JSON file from sources/translation_helps (cached).

    Returns an empty list if the file does not exist so callers can gracefully
    detect the absence of translation helps for a given book without raising.
    """
    path = data_root / f"{file_stem}.json"
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def _build_index(entries: List[Dict[str, Any]]) -> Dict[Tuple[int, int], Dict[str, Any]]:
    """Build a (chapter, verse) -> entry index from translation_helps entries."""
    idx: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for e in entries:
        ref = e.get("reference", "")
        parsed = parse_ch_verse_from_reference(ref)
        if parsed is None:
            continue
        idx[parsed] = e
    return idx


def _select_range(
    idx: Dict[Tuple[int, int], Dict[str, Any]],
    start_ch: int,
    start_vs: int | None,
    end_ch: int | None,
    end_vs: int | None,
) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    s_ch = start_ch
    s_vs = start_vs if start_vs is not None else 1
    e_ch = end_ch if end_ch is not None else s_ch
    e_vs = end_vs if end_vs is not None else 10_000
    for (ch, vs), entry in sorted(idx.items()):
        if (ch < s_ch) or (ch == s_ch and vs < s_vs):
            continue
        if (ch > e_ch) or (ch == e_ch and vs > e_vs):
            break
        items.append(entry)
    return items


def select_translation_helps(
    data_root: Path,
    canonical_book: str,
    ranges: Iterable[Tuple[int, int | None, int | None, int | None]],
) -> List[Dict[str, Any]]:
    """Select translation helps entries for a canonical book and ranges."""
    mapping = BOOK_MAP[canonical_book]
    entries = load_book_json(data_root, mapping["file_stem"])  # cached
    idx = _build_index(entries)
    out: List[Dict[str, Any]] = []
    for sc, sv, ec, ev in ranges:
        out.extend(_select_range(idx, sc, sv, ec, ev))
    return out


def _present_file_stems(data_root: Path) -> set[str]:
    """Return a set of available file stems under sources/translation_helps."""
    stems: set[str] = set()
    if not data_root.exists():
        return stems
    for p in data_root.glob("*.json"):
        stems.add(p.stem)
    return stems


def get_missing_th_books(data_root: Path) -> list[str]:
    """Return canonical book names that are missing from translation_helps.

    Compares canonical mapping stems to present files under the data_root.
    """
    present = _present_file_stems(data_root)
    missing: list[str] = []
    for canonical, mapping in BOOK_MAP.items():
        if mapping["file_stem"] not in present:
            missing.append(canonical)
    return missing
