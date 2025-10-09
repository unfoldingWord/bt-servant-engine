"""Unit tests for response helper utilities."""

from __future__ import annotations

from bt_servant_engine.services import response_helpers as rh

# pylint: disable=missing-function-docstring


def test_is_protected_response_item_detects_suppress_flag() -> None:
    assert rh.is_protected_response_item({"response": {"suppress_translation": True}}) is True
    assert rh.is_protected_response_item({"response": "plain text"}) is False

    scripture_segments = {
        "response": {
            "segments": [
                {"type": "scripture", "text": "John 3:16"},
                {"type": "plain", "text": "For God so loved..."},
            ]
        }
    }
    assert rh.is_protected_response_item(scripture_segments) is True


def test_partition_response_items_splits_groups() -> None:
    responses: list[dict[str, object]] = [
        {"response": {"suppress_translation": True}},
        {"response": "plain"},
        {"response": {"segments": [{"type": "scripture"}]}},
    ]
    protected, normal = rh.partition_response_items(responses)
    assert len(protected) == 2
    assert len(normal) == 1
    assert normal[0]["response"] == "plain"


def test_normalize_single_response_handles_str_and_dict() -> None:
    assert rh.normalize_single_response({"response": "simple"}) == "simple"
    item: dict[str, object] = {"response": {"segments": []}}
    assert rh.normalize_single_response(item) is item


def test_sample_for_language_detection_trims_and_snaps() -> None:
    # Leading whitespace trimmed
    assert rh.sample_for_language_detection("   hello world") == "hello world"
    # Sample truncated at whitespace boundary
    text = "word1 word2 word3"
    assert rh.sample_for_language_detection(text, sample_chars=10) == "word1"
    # When no whitespace in snippet, fall back to raw clipping
    assert rh.sample_for_language_detection("abcdefghijk", sample_chars=5) == "abcde"
