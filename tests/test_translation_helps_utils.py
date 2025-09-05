"""Unit tests for translation helps utilities.

Focus: graceful behavior when per-book translation helps files are missing.
"""
from pathlib import Path

from utils.translation_helps import select_translation_helps, get_missing_th_books


def test_select_translation_helps_missing_book_returns_empty_list() -> None:
    """Selecting helps for a book without a JSON file yields no entries."""
    # Numbers translation helps file (num.json) is not present in repo sources.
    # Expect graceful fallback to an empty list rather than raising.
    data_root = Path("sources") / "translation_helps"
    out = select_translation_helps(data_root, "Numbers", [(2, 1, 2, 5)])
    assert not out


def test_get_missing_th_books_includes_numbers() -> None:
    """Numbers is currently missing in the translation_helps dataset."""
    data_root = Path("sources") / "translation_helps"
    missing = get_missing_th_books(data_root)
    assert "Numbers" in missing
