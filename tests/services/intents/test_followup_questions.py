"""Tests for intent follow-up questions functionality."""


from bt_servant_engine.core.intents import IntentType
from bt_servant_engine.services.intents.followup_questions import (
    INTENT_FOLLOWUP_QUESTIONS,
    add_followup_if_needed,
    get_followup_for_intent,
)


class TestGetFollowupForIntent:
    """Test get_followup_for_intent function."""

    def test_returns_english_followup_for_retrieve_scripture(self):
        """Returns English follow-up for retrieve scripture intent."""
        result = get_followup_for_intent(IntentType.RETRIEVE_SCRIPTURE, "en")
        assert result == "Would you like to look up another Bible passage?"

    def test_returns_spanish_followup_for_retrieve_scripture(self):
        """Returns Spanish follow-up for retrieve scripture intent."""
        result = get_followup_for_intent(IntentType.RETRIEVE_SCRIPTURE, "es")
        assert result == "¿Le gustaría buscar otro pasaje bíblico?"

    def test_returns_english_fallback_for_unsupported_language(self):
        """Falls back to English when language not available."""
        result = get_followup_for_intent(IntentType.RETRIEVE_SCRIPTURE, "xyz")
        assert result == "Would you like to look up another Bible passage?"

    def test_returns_generic_fallback_for_undefined_intent(self):
        """Returns generic fallback for intent without specific follow-up."""
        # Use an intent that might not have a specific follow-up
        result = get_followup_for_intent(IntentType.CONVERSE_WITH_BT_SERVANT, "en")
        assert "help" in result.lower() or "assist" in result.lower()

    def test_all_defined_intents_have_english_translations(self):
        """All intents in INTENT_FOLLOWUP_QUESTIONS have English translations."""
        for intent_type, translations in INTENT_FOLLOWUP_QUESTIONS.items():
            assert "en" in translations, (
                f"Intent {intent_type} missing English translation"
            )
            assert translations["en"], (
                f"Intent {intent_type} has empty English translation"
            )

    def test_followup_questions_have_multiple_languages(self):
        """Follow-up questions support multiple languages."""
        # Check that at least English and Spanish are supported
        for intent_type, translations in INTENT_FOLLOWUP_QUESTIONS.items():
            assert "en" in translations, f"Intent {intent_type} missing English"
            assert "es" in translations, f"Intent {intent_type} missing Spanish"


class TestAddFollowupIfNeeded:
    """Test add_followup_if_needed function."""

    def test_adds_followup_when_not_already_added(self):
        """Adds follow-up question when flag is False."""
        state = {
            "followup_question_added": False,
            "user_response_language": "en",
        }
        response = "Here is your scripture passage."

        result, added = add_followup_if_needed(response, state, IntentType.RETRIEVE_SCRIPTURE)

        assert "Would you like to look up another Bible passage?" in result
        assert added is True

    def test_does_not_add_followup_when_already_added(self):
        """Does not add follow-up when flag is already True."""
        state = {
            "followup_question_added": True,
            "user_response_language": "en",
        }
        response = "Here is your scripture passage."

        result, added = add_followup_if_needed(response, state, IntentType.RETRIEVE_SCRIPTURE)

        # Should return unchanged
        assert result == response
        assert "Would you like" not in result
        assert added is False

    def test_uses_correct_language_from_state(self):
        """Uses user's preferred language from state."""
        state = {
            "followup_question_added": False,
            "user_response_language": "es",
        }
        response = "Aquí está su pasaje de las Escrituras."

        result, added = add_followup_if_needed(response, state, IntentType.RETRIEVE_SCRIPTURE)

        assert "¿Le gustaría buscar otro pasaje bíblico?" in result
        assert added is True

    def test_defaults_to_english_when_language_not_in_state(self):
        """Defaults to English when language not specified in state."""
        state = {"followup_question_added": False}
        response = "Here is your scripture passage."

        result, added = add_followup_if_needed(response, state, IntentType.RETRIEVE_SCRIPTURE)

        assert "Would you like to look up another Bible passage?" in result
        assert added is True

    def test_preserves_response_content(self):
        """Preserves original response content when adding follow-up."""
        state = {
            "followup_question_added": False,
            "user_response_language": "en",
        }
        response = "Original content here."

        result, added = add_followup_if_needed(response, state, IntentType.RETRIEVE_SCRIPTURE)

        assert result.startswith("Original content here.")
        assert "\n\n" in result  # Check for spacing
        assert added is True

    def test_different_intents_get_different_followups(self):
        """Different intent types get different follow-up questions."""
        state_scripture = {
            "followup_question_added": False,
            "user_response_language": "en",
        }
        state_translation = {
            "followup_question_added": False,
            "user_response_language": "en",
        }

        result_scripture, _ = add_followup_if_needed(
            "Response 1", state_scripture, IntentType.RETRIEVE_SCRIPTURE
        )
        result_translation, _ = add_followup_if_needed(
            "Response 2", state_translation, IntentType.GET_TRANSLATION_HELPS
        )

        assert "Bible passage" in result_scripture
        assert "translation question" in result_translation
        assert result_scripture != result_translation

    def test_translates_missing_language_and_caches(self):
        """Dynamically translates follow-up when language is missing and caches it."""
        state = {
            "followup_question_added": False,
            "user_response_language": "xx",
        }

        translations: list[tuple[str, str]] = []

        def fake_translate(text: str, lang: str) -> str:
            translations.append((text, lang))
            return f"{text}-{lang}"

        followups = INTENT_FOLLOWUP_QUESTIONS[IntentType.SET_RESPONSE_LANGUAGE]
        english_base = followups["en"]
        try:
            followups.pop("xx", None)
            result, added = add_followup_if_needed(
                "Content",
                state,
                IntentType.SET_RESPONSE_LANGUAGE,
                translate_text_fn=fake_translate,
            )
            assert added is True
            assert translations == [(english_base, "xx")]
            assert "-xx" in result
            assert followups.get("xx") == f"{english_base}-xx"
        finally:
            followups.pop("xx", None)


class TestFollowupQuestionsCoverage:
    """Test that follow-up questions cover expected intents."""

    def test_retrieve_scripture_has_followup(self):
        """RETRIEVE_SCRIPTURE intent has follow-up defined."""
        assert IntentType.RETRIEVE_SCRIPTURE in INTENT_FOLLOWUP_QUESTIONS

    def test_get_translation_helps_has_followup(self):
        """GET_TRANSLATION_HELPS intent has follow-up defined."""
        assert IntentType.GET_TRANSLATION_HELPS in INTENT_FOLLOWUP_QUESTIONS

    def test_set_response_language_has_followup(self):
        """SET_RESPONSE_LANGUAGE intent has follow-up defined."""
        assert IntentType.SET_RESPONSE_LANGUAGE in INTENT_FOLLOWUP_QUESTIONS

    def test_clear_response_language_has_followup(self):
        """CLEAR_RESPONSE_LANGUAGE intent has follow-up defined."""
        assert IntentType.CLEAR_RESPONSE_LANGUAGE in INTENT_FOLLOWUP_QUESTIONS

    def test_retrieve_system_information_has_followup(self):
        """RETRIEVE_SYSTEM_INFORMATION intent has follow-up defined."""
        assert IntentType.RETRIEVE_SYSTEM_INFORMATION in INTENT_FOLLOWUP_QUESTIONS

    def test_passage_related_intents_have_followups(self):
        """All passage-related intents have follow-ups defined."""
        passage_intents = [
            IntentType.GET_PASSAGE_SUMMARY,
            IntentType.GET_PASSAGE_KEYWORDS,
            IntentType.LISTEN_TO_SCRIPTURE,
            IntentType.TRANSLATE_SCRIPTURE,
        ]
        for intent in passage_intents:
            assert intent in INTENT_FOLLOWUP_QUESTIONS, f"{intent} missing follow-up"
