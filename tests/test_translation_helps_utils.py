"""Unit tests for translation helps utilities.

Focus: graceful behavior when per-book translation helps files are missing.
"""
from pathlib import Path

from utils.translation_helps import select_translation_helps


def test_select_translation_helps_missing_book_returns_empty_list() -> None:
    # Numbers translation helps file (num.json) is not present in repo sources.
    # Expect graceful fallback to an empty list rather than raising.
    data_root = Path("sources") / "translation_helps"
    out = select_translation_helps(data_root, "Numbers", [(2, 1, 2, 5)])
    assert out == []

