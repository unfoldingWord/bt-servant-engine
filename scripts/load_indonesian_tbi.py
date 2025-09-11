"""Generate Indonesian TBI verse JSONs from USFM.

# pylint: disable=duplicate-code

Outputs per-book files matching the BSB JSON format used by utils.bsb:

- Path: sources/bible_data/id/tbi/<stem>.json
- Format: array of {"reference": "Gen 1:1", "text": "..."}

USFM source:
https://github.com/unfoldingWord/bt-servant-engine-data-loaders/tree/main/datasets/tbi
"""

from __future__ import annotations
from pathlib import Path
from typing import List

from utils.usfm_to_json import build_dataset


def main(argv: List[str] | None = None) -> int:  # pylint: disable=unused-argument
    """Build Indonesian TBI per-book JSON files from USFM input.

    Uses default paths suitable for local builds. Pass argv to conform to
    script entry expectations; currently unused.
    """
    src_dir: Path = Path("/tmp/bt-loaders/datasets/tbi")
    out_root: Path = Path("sources") / "bible_data" / "id" / "tbi"
    build_dataset(src_dir, out_root)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
