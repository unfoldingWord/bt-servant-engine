"""OpenAI-backed tests for intent classification.

Calls brain.determine_intents with real OpenAI to verify summary vs
keywords detection across a variety of phrasings and tricky books.
"""
# pylint: disable=missing-function-docstring,line-too-long,wrong-import-order,duplicate-code
from __future__ import annotations

from dotenv import load_dotenv

from typing import Any, cast

import pytest
from brain import determine_intents, IntentType
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
    "query",
    [
        "summarize 3 John",
        "Please summarize John 3:16-18",
        "Give me a summary of John 1–4",
        "Summarize 1 Peter",
        "A summary of 2 John would be helpful",
    ],
)
def test_intents_detect_summary(query: str) -> None:
    state: dict[str, Any] = {"transformed_query": query}
    out = determine_intents(cast(Any, state))
    intents = set(out["user_intents"])  # list[IntentType]
    assert IntentType.GET_PASSAGE_SUMMARY in intents


@pytest.mark.parametrize(
    "query",
    [
        "What are the keywords in 3 John?",
        "List key words in Genesis 1",
        "Important words in John 3:16",
        "Give keywords for 2 Peter 1",
        "Keywords for 1 Thessalonians 4",
    ],
)
def test_intents_detect_keywords(query: str) -> None:
    state: dict[str, Any] = {"transformed_query": query}
    out = determine_intents(cast(Any, state))
    intents = set(out["user_intents"])  # list[IntentType]
    assert IntentType.GET_PASSAGE_KEYWORDS in intents


@pytest.mark.parametrize(
    "query",
    [
        "How do I translate the Bible?",
        "What are the steps of the FIA process?",
        "What does FIA step 2 look like in the first chapter of Mark?",
    ],
)
def test_intents_detect_consult_fia_resources(query: str) -> None:
    state: dict[str, Any] = {"transformed_query": query}
    out = determine_intents(cast(Any, state))
    intents = set(out["user_intents"])  # list[IntentType]
    assert IntentType.CONSULT_FIA_RESOURCES in intents
