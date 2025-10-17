"""Tests for deterministic passage follow-up helpers."""

from bt_servant_engine.core.intents import IntentType
from bt_servant_engine.services.intents.translation_intents import (
    TranslationContext,
    _wrap_translation_response,
)
from bt_servant_engine.services.passage_followups import (
    build_followup_question,
    propose_next_passage_range,
)


class TestProposeNextPassageRange:
    """Tests for propose_next_passage_range."""

    def test_within_same_chapter(self) -> None:
        """Returns next verses within the same chapter when available."""
        result = propose_next_passage_range(
            "John",
            [(3, 16, 3, 20)],
            verse_limit=5,
        )
        assert result is not None
        next_book, next_range = result
        assert next_book == "John"
        assert next_range == (3, 21, 25)

    def test_rolls_to_next_chapter(self) -> None:
        """Moves to the next chapter when the current chapter is exhausted."""
        result = propose_next_passage_range(
            "John",
            [(3, 33, 3, 36)],
            verse_limit=5,
        )
        assert result is not None
        next_book, next_range = result
        assert next_book == "John"
        assert next_range == (4, 1, 5)

    def test_wraps_to_genesis_after_revelation(self) -> None:
        """Wraps around to Genesis when Revelation is exhausted."""
        result = propose_next_passage_range(
            "Revelation",
            [(22, 17, 22, 21)],
            verse_limit=5,
        )
        assert result is not None
        next_book, next_range = result
        assert next_book == "Genesis"
        assert next_range == (1, 1, 5)


class TestBuildFollowupQuestion:  # pylint: disable=too-few-public-methods
    """Tests for building localized follow-up questions."""

    def test_builds_english_question(self) -> None:
        """Returns an English follow-up question with the proposed label."""
        context = {
            "intent": IntentType.GET_PASSAGE_SUMMARY,
            "book": "John",
            "ranges": [(3, 16, 3, 20)],
        }
        question = build_followup_question(
            IntentType.GET_PASSAGE_SUMMARY,
            context,
            target_language="en",
        )
        assert question is not None
        assert "John 3:21-25" in question

    def test_translation_followup_includes_target_language(self) -> None:
        """Translation follow-up mentions the resolved target language."""
        context = {
            "intent": IntentType.TRANSLATE_SCRIPTURE,
            "book": "John",
            "ranges": [(1, 1, 1, 1)],
            "target_language": "es",
        }
        question = build_followup_question(
            IntentType.TRANSLATE_SCRIPTURE,
            context,
            target_language="en",
        )
        assert question == "Would you like me to translate John 1:2-6 into Spanish?"


def test_translation_followup_context_carries_target_language() -> None:
    """Translation responses expose the target language for follow-up generation."""
    context = TranslationContext(
        canonical_book="John",
        ranges=[(1, 1, 1, 1)],
        target_code="es",
        resolved_lang="en",
        body_source="In the beginning...",
        header_suffix="1:1",
        verses=[("1:1", "In the beginning...")],
    )
    response = _wrap_translation_response(context, {"segments": []})
    assert response["passage_followup_context"]["target_language"] == "es"
