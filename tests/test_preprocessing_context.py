"""Unit tests for structured intent context extraction."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from bt_servant_engine.core.intents import IntentType, IntentWithContext, UserIntentsStructured
from bt_servant_engine.services.preprocessing import determine_intents_structured


def _mock_parse_result(items: list[IntentWithContext]) -> MagicMock:
    mock_response = MagicMock()
    mock_response.output_parsed = UserIntentsStructured(intents=items)
    mock_response.usage = None
    return mock_response


def test_structured_extraction_respects_expected_intents(caplog: pytest.LogCaptureFixture) -> None:
    """Structured extraction should ignore hallucinated intents and fall back to query text."""
    client = MagicMock()
    model_items = [
        IntentWithContext(
            intent=IntentType.CONSULT_FIA_RESOURCES,
            context_text="Can you tell me how the FIA process works?",
        ),
        IntentWithContext(
            intent=IntentType.GET_PASSAGE_SUMMARY,
            context_text="Tell me about Barnabas from the Bible.",
        ),
    ]
    client.responses.parse.return_value = _mock_parse_result(model_items)

    query = (
        "Can you tell me how the FIA process works and also tell me about Barnabas from the Bible?"
    )
    expected = [
        IntentType.CONSULT_FIA_RESOURCES,
        IntentType.GET_BIBLE_TRANSLATION_ASSISTANCE,
    ]

    caplog.set_level("WARNING")
    results = determine_intents_structured(client, query, expected)

    assert [item.intent for item in results] == expected
    assert results[0].context_text == "Can you tell me how the FIA process works?"
    assert results[1].context_text == query.strip()
    assert any("unexpected intent" in message for message in caplog.messages)


def test_structured_extraction_falls_back_on_exception() -> None:
    """If the structured call fails entirely, fall back to full query context."""
    client = MagicMock()
    client.responses.parse.side_effect = RuntimeError("boom")

    query = "Summarize John 1"
    expected = [IntentType.GET_PASSAGE_SUMMARY]

    results = determine_intents_structured(client, query, expected)

    assert len(results) == 1
    assert results[0].intent == IntentType.GET_PASSAGE_SUMMARY
    assert results[0].context_text == query


def test_structured_extraction_with_no_expected_intents() -> None:
    """When no expected intents are supplied, an empty list is returned."""
    client = MagicMock()
    results = determine_intents_structured(client, "anything", [])
    assert results == []
