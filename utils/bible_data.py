"""Bible data path resolution utilities.

Resolves `sources/bible_data/<lang>/<version>` roots with sensible fallbacks:
- response_language (if set) → query_language → en
- Language aliasing: map variants like `pt-BR`, `pt-PT` → `pt`.
- Ensures the selected root exists and contains JSON files.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Dict
import json

_BIBLE_DATA_ROOT = Path("sources") / "bible_data"


def _alias_lang(code: Optional[str]) -> Optional[str]:
    if not code:
        return None
    lc = str(code).strip().lower()
    if lc in {"pt-br", "pt_pt", "pt-pt", "ptbr"}:
        return "pt"
    return lc


def list_available_sources() -> list[tuple[str, str]]:
    """Return available (lang, version) pairs by scanning the filesystem."""
    out: list[tuple[str, str]] = []
    if not _BIBLE_DATA_ROOT.exists():
        return out
    for lang_dir in _BIBLE_DATA_ROOT.iterdir():
        if not lang_dir.is_dir():
            continue
        for ver_dir in lang_dir.iterdir():
            if not ver_dir.is_dir():
                continue
            if any(ver_dir.glob("*.json")):
                out.append((lang_dir.name, ver_dir.name))
    return sorted(out)


def _has_json(dir_path: Path) -> bool:
    return dir_path.exists() and any(dir_path.glob("*.json"))


def resolve_bible_data_root(
    response_language: Optional[str],
    query_language: Optional[str],
    requested_lang: Optional[str] = None,
    requested_version: Optional[str] = None,
    ) -> Tuple[Path, str, str]:
    """Resolve the best bible data root and return (path, lang, version).

    Precedence:
    - If a specific `requested_lang` is provided, try that first.
    - Then try response_language, then query_language, then 'en'.

    Version handling:
    - If `requested_version` is provided, prefer it.
    - Otherwise, select the first available version under the chosen language.

    Raises FileNotFoundError if no suitable root is found or if the final
    fallback `en/<any>` is missing/empty.
    """
    candidates: list[str] = []
    for val in (requested_lang, response_language, query_language, "en"):
        a = _alias_lang(val)
        if a and a not in candidates:
            candidates.append(a)

    # Try requested version first if given; otherwise pick any present version.
    for lang in candidates:
        lang_dir = _BIBLE_DATA_ROOT / lang
        if not lang_dir.exists():
            continue
        versions: list[Path] = []
        if requested_version:
            versions = [lang_dir / requested_version]
        else:
            versions = [p for p in lang_dir.iterdir() if p.is_dir()]
        for vdir in versions:
            if _has_json(vdir):
                return vdir, lang, vdir.name

    # Hard fail if even en is not found
    raise FileNotFoundError(
        "No bible data found. Expected at least sources/bible_data/en/<version> "
        "(e.g., en/bsb) with JSON files."
    )


def load_book_titles(data_root: Path) -> Dict[str, str]:
    """Load localized book-title mapping from a dataset root, if present.

    Returns a mapping of canonical English book name → localized title (full name).
    """
    p = data_root / "_book_titles.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - tolerate malformed files  # pylint: disable=broad-except
        return {}
