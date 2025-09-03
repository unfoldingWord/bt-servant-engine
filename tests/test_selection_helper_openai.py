"""OpenAI-backed contract tests for passage selection parsing.

These tests call the real OpenAI Responses API via the helper
`brain._resolve_selection_for_single_book`. They will be skipped when
OPENAI_API_KEY is not configured for networked runs.
"""
# pylint: disable=missing-function-docstring,line-too-long,duplicate-code
from __future__ import annotations

from dotenv import load_dotenv
import pytest

from brain import _resolve_selection_for_single_book, Language
from config import config as app_config


def _has_real_openai() -> bool:
    key = str(app_config.OPENAI_API_KEY)
    return bool(key and key != "test" and key.startswith("sk-"))


load_dotenv(override=True)

pytestmark = [
    pytest.mark.openai,
    pytest.mark.skipif(not _has_real_openai(), reason="OPENAI_API_KEY not set for live OpenAI tests"),
]


@pytest.mark.parametrize(
    "query,expect_book,expect_range_kind",
    [
        ("3 John", "3 John", "whole_book"),
        ("2 John", "2 John", "whole_book"),
        ("1 Peter", "1 Peter", "whole_book"),
        ("John 3", "John", "whole_chapter"),
        ("John 3:16-18", "John", "same_chapter_range"),
        ("John 3:16–4:2", "John", "cross_chapter_range"),
        ("Genesis 1–4", "Genesis", "multi_chapter"),
        ("Titus", "Titus", "whole_book"),
    ],
)
def test_selection_parsing_varied_queries(query: str, expect_book: str, expect_range_kind: str) -> None:
    book, ranges, err = _resolve_selection_for_single_book(query, Language.ENGLISH.value)
    assert err is None, f"unexpected error for {query}: {err}"
    assert book == expect_book
    assert ranges and isinstance(ranges, list)
    sc, sv, ec, ev = ranges[0]
    if expect_range_kind == "whole_book":
        assert sc == 1 and sv is None and (ec is not None and ec >= 10_000) and ev is None
    elif expect_range_kind == "whole_chapter":
        assert sc >= 1 and sv is None and (ec is None or ec == sc) and ev is None
    elif expect_range_kind == "same_chapter_range":
        assert sc >= 1 and sv is not None and (ec is None or ec == sc) and ev is not None
    elif expect_range_kind == "cross_chapter_range":
        assert sc >= 1 and sv is not None and (ec is not None and ec > sc)
    elif expect_range_kind == "multi_chapter":
        assert sc >= 1 and sv is None and (ec is not None and ec >= sc) and ev is None


def test_selection_multiple_books_returns_guidance_message() -> None:
    # Ambiguous cross-book input should not fabricate a selection
    query = "Gen–Exo"
    book, ranges, err = _resolve_selection_for_single_book(query, Language.ENGLISH.value)
    assert book is None and ranges is None
    assert isinstance(err, str) and ("choose one book" in err or "one book" in err)
