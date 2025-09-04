#!/usr/bin/env python3
"""
Builds the Translation Academy (TA) dataset (Dataset 2).

Process:
- Scan all JSON under `sources/translation_helps/` and collect unique
  `support_reference` values from each note.
- Normalize each reference by removing the `rc://<lang>/ta/man` prefix to
  yield a stem like `/translate/translate-names`.
- For each stem, fetch three markdown files from Door43 (raw):
  - title.md, sub-title.md, 01.md
  - Base URL: https://git.door43.org/unfoldingWord/en_ta/raw/branch/master
  - Full path: <base><stem>/<file>
- Write JSON to `sources/ta_data/<stem>.json` with shape:
  {"title": ..., "sub-title": ..., "text": ...}

Logs warnings when:
- Door43 is unreachable.
- A specific path (stem) is missing.
- One or more expected files are missing for a stem.

Usage:
  python scripts/build_ta_data.py
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import httpx


REPO_ROOT = Path(__file__).resolve().parents[1]
TH_DIR = REPO_ROOT / "sources" / "translation_helps"
TA_OUT_DIR = REPO_ROOT / "sources" / "ta_data"


PREFIX_RE = re.compile(r"^rc://[^/]+/ta/man(?P<stem>/.*)$")
DOOR43_BASE = "https://git.door43.org/unfoldingWord/en_ta/raw/branch/master"


def _normalize_support_reference(ref: str) -> Optional[str]:
    """Convert `rc://<lang>/ta/man/<path>` to `/<path>`; return None if not match."""
    if not ref or not ref.startswith("rc://"):
        return None
    m = PREFIX_RE.match(ref)
    return m.group("stem") if m else None


def _collect_ta_stems() -> dict[str, str]:
    """Collect mapping of TA stem -> original support_reference.

    If multiple support_reference values map to the same stem, prefer the
    variant that contains the wildcard language segment ("rc://*/...") to
    maximize alignment with the translation_helps dataset.
    """
    stems: dict[str, str] = {}
    for path in sorted(TH_DIR.glob("*.json")):
        with path.open("r", encoding="utf-8") as f:
            try:
                items = json.load(f)
            except json.JSONDecodeError as exc:  # noqa: TRY003
                print(f"[warn] Skipping invalid JSON: {path}: {exc}")
                continue
        for obj in items:
            for note in obj.get("notes", []):
                sr = note.get("support_reference")
                stem = _normalize_support_reference(sr) if isinstance(sr, str) else None
                if stem and isinstance(sr, str):
                    if stem not in stems:
                        stems[stem] = sr
                    else:
                        # Prefer wildcard version if present
                        existing = stems[stem]
                        if "rc://*/" in sr and "rc://*/" not in existing:
                            stems[stem] = sr
    return stems


@dataclass(frozen=True)
class TAEntry:
    """Container for a single TA article."""

    title: str
    sub_title: str
    text: str
    support_reference: str

    def to_json_obj(self) -> dict:
        """Serialize to the required shape."""
        return {
            "title": self.title,
            "sub-title": self.sub_title,
            "text": self.text,
            "support_reference": self.support_reference,
        }


def _fetch_text(client: httpx.Client, url: str) -> Optional[str]:
    try:
        r = client.get(url, timeout=20)
        if r.status_code == 200:
            return r.text
        return None
    except httpx.HTTPError:
        return None


def _build_one(
    client: httpx.Client, stem: str, support_reference: str
) -> tuple[Optional[TAEntry], list[str]]:
    base = f"{DOOR43_BASE}{stem}"
    errs: list[str] = []
    title = _fetch_text(client, f"{base}/title.md")
    sub_title = _fetch_text(client, f"{base}/sub-title.md")
    text = _fetch_text(client, f"{base}/01.md")
    pairs = (("title.md", title), ("sub-title.md", sub_title), ("01.md", text))
    missing = [name for name, val in pairs if val is None]
    if missing:
        errs.append(f"missing files for {stem}: {', '.join(missing)}")
        return None, errs
    return (
        TAEntry(
            title=title or "",
            sub_title=sub_title or "",
            text=text or "",
            support_reference=support_reference,
        ),
        errs,
    )


def build_ta_dataset() -> None:
    """Build TA dataset into `sources/ta_data` from collected stems."""
    TA_OUT_DIR.mkdir(parents=True, exist_ok=True)
    stems_map = _collect_ta_stems()
    if not stems_map:
        print("[warn] No TA stems collected from translation_helps; nothing to do")
        return
    print(f"[info] Collected {len(stems_map)} unique TA stems")

    # Door43 availability check
    try:
        httpx.get(DOOR43_BASE, timeout=10)
    except httpx.HTTPError:
        print("[error] Cannot access Door43 base; aborting")
        return

    created = 0
    errors = 0
    with httpx.Client(headers={"User-Agent": "bt-servant/ta-builder"}) as client:
        for stem in sorted(stems_map.keys()):
            support_reference = stems_map[stem]
            entry, errs = _build_one(client, stem, support_reference)
            if errs:
                errors += 1
                print("[warn] " + "; ".join(errs))
                continue

            # Write to nested path: sources/ta_data/<stem>.json
            # Example: /translate/translate-names ->
            #   sources/ta_data/translate/translate-names.json
            rel_path = stem.lstrip("/") + ".json"
            out_path = TA_OUT_DIR / rel_path
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with out_path.open("w", encoding="utf-8") as f:
                assert entry is not None  # for type checkers
                f.write(
                    json.dumps(
                        entry.to_json_obj(), ensure_ascii=False, indent=2
                    )
                )
                f.write("\n")
            created += 1
            print(f"[ok] Wrote {out_path}")

    print(f"[done] Created: {created}; with warnings: {errors}")


if __name__ == "__main__":
    build_ta_dataset()
