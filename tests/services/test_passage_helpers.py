"""Unit tests for passage helper utilities."""

from __future__ import annotations

from bt_servant_engine.services import passage_helpers as helpers

# pylint: disable=missing-function-docstring

BOOK_MAP = {
    "Genesis": {"ref_abbr": "Gen"},
    "John": {"ref_abbr": "Jn"},
}


def test_book_patterns_include_canonical_and_abbr() -> None:
    patterns = helpers.book_patterns(BOOK_MAP)
    canonical_entries = [p for p in patterns if p[0] == "Genesis"]
    assert any(r"\bGenesis\b" in regex for _, regex in canonical_entries)
    assert any(r"\bGen\b" in regex for _, regex in canonical_entries)


def test_detect_mentioned_books_orders_and_dedupes() -> None:
    text = "Genesis sets the stage, but John 3 offers insight."
    detected = helpers.detect_mentioned_books(text, BOOK_MAP)
    assert detected == ["Genesis", "John"]


def test_choose_primary_book_prefers_digits_nearby() -> None:
    text = "Read John 3 before turning back to Genesis."
    primary = helpers.choose_primary_book(text, ["Genesis", "John"], BOOK_MAP)
    assert primary == "John"

    fallback = helpers.choose_primary_book("Mention Genesis and John", ["Genesis", "John"], BOOK_MAP)
    assert fallback is None
