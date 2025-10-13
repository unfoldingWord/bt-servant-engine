"""Integration tests for intent follow-up questions in translate_responses."""

from unittest.mock import patch


from bt_servant_engine.core.intents import IntentType
from bt_servant_engine.services.brain_nodes import translate_responses


class TestFollowupIntegration:
    """Test follow-up questions integration with translate_responses."""

    def test_adds_followup_for_single_intent(self):
        """Adds intent-specific follow-up for single intent response."""
        state = {
            "responses": [
                {
                    "intent": IntentType.RETRIEVE_SCRIPTURE,
                    "response": "Here is John 3:16",
                }
            ],
            "user_response_language": "en",
            "query_language": "en",
            "agentic_strength": "normal",
            "followup_question_added": False,
        }

        with patch(
            "bt_servant_engine.services.continuation_prompts.generate_continuation_prompt"
        ) as mock_continuation:
            mock_continuation.return_value = None  # No queued intents

            result = translate_responses(state)

            assert len(result["translated_responses"]) == 1
            response = result["translated_responses"][0]
            assert "Would you like to look up another Bible passage?" in response
            assert state["followup_question_added"] is True

    def test_multi_intent_followup_takes_precedence(self):
        """Multi-intent continuation prompt takes precedence over intent-specific follow-up."""
        state = {
            "responses": [
                {
                    "intent": IntentType.RETRIEVE_SCRIPTURE,
                    "response": "Here is John 3:16",
                }
            ],
            "user_id": "test_user",
            "user_response_language": "en",
            "query_language": "en",
            "agentic_strength": "normal",
            "followup_question_added": False,
        }

        continuation_prompt = "\n\nWould you like me to continue with your next request?"

        with patch(
            "bt_servant_engine.services.continuation_prompts.generate_continuation_prompt"
        ) as mock_continuation:
            mock_continuation.return_value = continuation_prompt

            result = translate_responses(state)

            assert len(result["translated_responses"]) == 1
            response = result["translated_responses"][0]

            # Should have continuation prompt, NOT intent-specific follow-up
            assert "continue with your next request" in response
            assert "Would you like to look up another Bible passage?" not in response
            assert state["followup_question_added"] is True

    def test_no_followup_for_converse_intent(self):
        """Does not add follow-up for CONVERSE_WITH_BT_SERVANT intent."""
        state = {
            "responses": [
                {
                    "intent": IntentType.CONVERSE_WITH_BT_SERVANT,
                    "response": "Hello! How can I help you?",
                }
            ],
            "user_response_language": "en",
            "query_language": "en",
            "agentic_strength": "normal",
            "followup_question_added": False,
        }

        with patch(
            "bt_servant_engine.services.continuation_prompts.generate_continuation_prompt"
        ) as mock_continuation:
            mock_continuation.return_value = None

            result = translate_responses(state)

            response = result["translated_responses"][0]
            # Should not add any follow-up question markers
            assert "?" not in response or response == "Hello! How can I help you?"
            # Flag should remain False since we explicitly skip these intents
            assert not state.get("followup_question_added", False)

    def test_no_followup_for_help_intent(self):
        """Does not add follow-up for RETRIEVE_SYSTEM_INFORMATION (help) intent (has its own)."""
        state = {
            "responses": [
                {
                    "intent": IntentType.RETRIEVE_SYSTEM_INFORMATION,
                    "response": "Here's how I can help...",
                }
            ],
            "user_response_language": "en",
            "query_language": "en",
            "agentic_strength": "normal",
            "followup_question_added": False,
        }

        with patch(
            "bt_servant_engine.services.continuation_prompts.generate_continuation_prompt"
        ) as mock_continuation:
            mock_continuation.return_value = None

            translate_responses(state)

            # Should not add intent-specific follow-up
            assert not state.get("followup_question_added", False)

    def test_followup_uses_user_language(self):
        """Follow-up question uses user's response language."""
        state = {
            "responses": [
                {
                    "intent": IntentType.RETRIEVE_SCRIPTURE,
                    "response": "Aquí está Juan 3:16",
                }
            ],
            "user_response_language": "es",
            "query_language": "en",
            "agentic_strength": "normal",
            "followup_question_added": False,
        }

        with patch(
            "bt_servant_engine.services.continuation_prompts.generate_continuation_prompt"
        ) as mock_continuation:
            mock_continuation.return_value = None

            result = translate_responses(state)

            response = result["translated_responses"][0]
            # Should have Spanish follow-up
            assert "¿Le gustaría buscar otro pasaje bíblico?" in response

    def test_different_intents_get_different_followups(self):
        """Different intent types receive appropriate follow-up questions."""
        intents_and_keywords = [
            (IntentType.RETRIEVE_SCRIPTURE, "Bible passage"),
            (IntentType.GET_TRANSLATION_HELPS, "translation question"),
            (IntentType.SET_RESPONSE_LANGUAGE, "What else"),
        ]

        for intent, keyword in intents_and_keywords:
            state = {
                "responses": [
                    {
                        "intent": intent,
                        "response": "Test response",
                    }
                ],
                "user_response_language": "en",
                "query_language": "en",
                "agentic_strength": "normal",
                "followup_question_added": False,
            }

            with patch(
                "bt_servant_engine.services.continuation_prompts.generate_continuation_prompt"
            ) as mock_continuation:
                mock_continuation.return_value = None

                result = translate_responses(state)
                response = result["translated_responses"][0]

                assert keyword in response, f"Intent {intent} missing expected keyword '{keyword}'"


class TestFollowupWithMultipleResponses:  # pylint: disable=too-few-public-methods
    """Test follow-up behavior with multiple responses."""

    def test_followup_appended_to_last_response_only(self):
        """Follow-up is appended only to the last response.

        NOTE: This test has multiple responses which doesn't happen in practice with
        sequential intent processing, but it validates the follow-up logic applies to
        the last response only.
        """
        state = {
            "responses": [
                {
                    "intent": IntentType.RETRIEVE_SCRIPTURE,
                    "response": "First response",
                },
                {
                    "intent": IntentType.RETRIEVE_SCRIPTURE,
                    "response": "Second response",
                },
            ],
            "user_response_language": "en",
            "query_language": "en",
            "agentic_strength": "normal",
            "followup_question_added": False,
        }

        with patch(
            "bt_servant_engine.services.continuation_prompts.generate_continuation_prompt"
        ) as mock_continuation:
            mock_continuation.return_value = None

            result = translate_responses(state)

            assert len(result["translated_responses"]) == 2
            # First response should not have follow-up
            assert "Would you like" not in result["translated_responses"][0]
            # Last response should have follow-up
            assert "Would you like to look up another Bible passage?" in result["translated_responses"][1]
