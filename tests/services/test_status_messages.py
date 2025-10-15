"""Tests for status message localization system."""
# pylint: disable=protected-access  # Tests need to access internal implementation

from unittest.mock import MagicMock, Mock, patch

import pytest

from bt_servant_engine.core.language import LANGUAGE_UNKNOWN
from bt_servant_engine.services import status_messages


class TestGetEffectiveResponseLanguage:
    """Test language resolution logic."""

    def test_returns_user_response_language_when_set(self):
        """Priority 1: User's explicit preference takes precedence."""
        state = {
            "user_response_language": "fr",
            "query_language": "es",
        }
        result = status_messages.get_effective_response_language(state)
        assert result == "fr"

    def test_returns_query_language_when_no_user_preference(self):
        """Priority 2: Detected query language when no user preference."""
        state = {
            "user_response_language": None,
            "query_language": "es",
        }
        result = status_messages.get_effective_response_language(state)
        assert result == "es"

    def test_returns_english_when_query_language_unknown(self):
        """Priority 3: English fallback when language is unknown."""
        state = {
            "user_response_language": None,
            "query_language": LANGUAGE_UNKNOWN,
        }
        result = status_messages.get_effective_response_language(state)
        assert result == "en"

    def test_returns_english_when_no_language_info(self):
        """English fallback when no language information available."""
        state = {}
        result = status_messages.get_effective_response_language(state)
        assert result == "en"

    def test_normalizes_language_code(self):
        """Language codes are normalized (lowercased and stripped)."""
        state = {
            "user_response_language": "  FR  ",
        }
        result = status_messages.get_effective_response_language(state)
        assert result == "fr"


class TestGetStatusMessage:
    """Test status message retrieval."""

    def test_returns_english_message_for_english(self):
        """Returns English text when language is English."""
        state = {"user_response_language": "en"}
        result = status_messages.get_status_message(
            status_messages.THINKING_ABOUT_MESSAGE, state
        )
        assert result == "Give me a few moments to think about your message."

    def test_returns_english_message_by_default(self):
        """Returns English text when no language specified."""
        state = {}
        result = status_messages.get_status_message(
            status_messages.TRANSCRIBING_VOICE, state
        )
        assert result == "I'm transcribing your voice message. Give me a moment."

    def test_all_message_keys_have_english_text(self):
        """All defined message keys have English translations."""
        state = {"user_response_language": "en"}
        message_keys = [
            status_messages.THINKING_ABOUT_MESSAGE,
            status_messages.SEARCHING_BIBLE_RESOURCES,
            status_messages.REVIEWING_FIA_GUIDANCE,
            status_messages.CHECKING_CAPABILITIES,
            status_messages.GENERATING_HELP_RESPONSE,
            status_messages.GATHERING_PASSAGE_SUMMARY,
            status_messages.EXTRACTING_KEYWORDS,
            status_messages.COMPILING_TRANSLATION_HELPS,
            status_messages.GATHERING_PASSAGE_TEXT,
            status_messages.PREPARING_AUDIO,
            status_messages.TRANSLATING_PASSAGE,
            status_messages.TRANSLATING_RESPONSE,
            status_messages.FINALIZING_RESPONSE,
            status_messages.TRANSCRIBING_VOICE,
            status_messages.PACKAGING_VOICE_RESPONSE,
            status_messages.PROCESSING_ERROR,
            status_messages.FOUND_RELEVANT_DOCUMENTS,
        ]
        for key in message_keys:
            result = status_messages.get_status_message(key, state)
            assert isinstance(result, str)
            assert len(result) > 0
            assert not result.startswith("[Unknown")

    def test_returns_error_for_unknown_message_key(self):
        """Returns error message for unknown message keys."""
        state = {"user_response_language": "en"}
        result = status_messages.get_status_message("INVALID_KEY", state)
        assert "[Unknown message: INVALID_KEY]" in result

    @patch("bt_servant_engine.services.status_messages._translate_dynamically")
    def test_calls_dynamic_translation_for_missing_language(self, mock_translate):
        """Calls dynamic translation when language not in pre-loaded data."""
        mock_translate.return_value = "Übersetzter Text"
        state = {"user_response_language": "de"}  # German not pre-loaded

        result = status_messages.get_status_message(
            status_messages.THINKING_ABOUT_MESSAGE, state
        )

        mock_translate.assert_called_once_with(
            status_messages.THINKING_ABOUT_MESSAGE, "de"
        )
        assert result == "Übersetzter Text"


class TestProgressMessages:
    """Tests for structured progress message helpers."""

    def test_thinking_message_uses_brain_emoji(self):
        """Brain status message includes the 🧠 emoji override."""
        state = {"user_response_language": "en"}
        result = status_messages.get_progress_message(
            status_messages.THINKING_ABOUT_MESSAGE, state
        )
        assert result["text"] == "Give me a few moments to think about your message."
        assert result["emoji"] == "🧠"

    def test_default_progress_message_uses_config_emoji(self, monkeypatch: pytest.MonkeyPatch):
        """Default emoji falls back to the configured global symbol."""
        state = {"user_response_language": "en"}
        monkeypatch.setattr(status_messages.config, "PROGRESS_MESSAGE_EMOJI", "⏳")
        monkeypatch.setattr(status_messages.config, "PROGRESS_MESSAGE_EMOJI_OVERRIDES", {})

        result = status_messages.get_progress_message(
            status_messages.TRANSCRIBING_VOICE, state
        )
        assert result["emoji"] == "⏳"

    def test_config_override_takes_precedence(self, monkeypatch: pytest.MonkeyPatch):
        """Configuration overrides take precedence over built-in defaults."""
        state = {"user_response_language": "en"}
        monkeypatch.setattr(status_messages.config, "PROGRESS_MESSAGE_EMOJI", "⏳")
        monkeypatch.setattr(
            status_messages.config,
            "PROGRESS_MESSAGE_EMOJI_OVERRIDES",
            {status_messages.THINKING_ABOUT_MESSAGE: "✨"},
        )

        result = status_messages.get_progress_message(
            status_messages.THINKING_ABOUT_MESSAGE, state
        )
        assert result["emoji"] == "✨"


class TestDynamicTranslation:
    """Test dynamic translation functionality."""

    @patch("bt_servant_engine.services.status_messages._get_openai_client")
    def test_translates_using_openai(self, mock_get_client):
        """Uses OpenAI to translate when language not pre-loaded."""
        # Clear cache to ensure fresh translation
        status_messages._DYNAMIC_TRANSLATION_CACHE.clear()

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Mensaje traducido al español"
        mock_client.chat.completions.create.return_value = mock_response

        result = status_messages._translate_dynamically(
            status_messages.THINKING_ABOUT_MESSAGE, "es"
        )

        assert result == "Mensaje traducido al español"
        mock_client.chat.completions.create.assert_called_once()

    @patch("bt_servant_engine.services.status_messages._get_openai_client")
    def test_caches_dynamic_translations(self, mock_get_client):
        """Caches dynamically translated messages to avoid repeated API calls."""
        # Clear cache to ensure fresh translation
        status_messages._DYNAMIC_TRANSLATION_CACHE.clear()

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Cached translation"
        mock_client.chat.completions.create.return_value = mock_response

        # First call - should hit OpenAI
        result1 = status_messages._translate_dynamically(
            status_messages.THINKING_ABOUT_MESSAGE, "it"
        )
        assert result1 == "Cached translation"
        assert mock_client.chat.completions.create.call_count == 1

        # Second call - should use cache
        result2 = status_messages._translate_dynamically(
            status_messages.THINKING_ABOUT_MESSAGE, "it"
        )
        assert result2 == "Cached translation"
        assert mock_client.chat.completions.create.call_count == 1  # No additional calls

    @patch("bt_servant_engine.services.status_messages._get_openai_client")
    def test_handles_openai_error_gracefully(self, mock_get_client):
        """Falls back to English when OpenAI translation fails."""
        # Clear cache
        status_messages._DYNAMIC_TRANSLATION_CACHE.clear()

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Exception("API Error")

        result = status_messages._translate_dynamically(
            status_messages.THINKING_ABOUT_MESSAGE, "de"
        )

        # Should return English fallback
        assert result == "Give me a few moments to think about your message."

    def test_returns_english_for_unknown_message_key(self):
        """Returns empty string when message key doesn't exist."""
        # Clear cache
        status_messages._DYNAMIC_TRANSLATION_CACHE.clear()

        result = status_messages._translate_dynamically("NONEXISTENT_KEY", "fr")

        assert result == ""


class TestOpenAIClientInitialization:  # pylint: disable=too-few-public-methods
    """Test lazy OpenAI client initialization."""

    @patch("bt_servant_engine.services.status_messages.OpenAI")
    def test_lazy_initializes_openai_client(self, mock_openai_class):
        """OpenAI client is lazily initialized on first use."""
        # Reset global client
        status_messages._openai_client = None

        mock_client_instance = MagicMock()
        mock_openai_class.return_value = mock_client_instance

        client1 = status_messages._get_openai_client()
        assert client1 == mock_client_instance
        mock_openai_class.assert_called_once()

        # Second call should return same instance
        client2 = status_messages._get_openai_client()
        assert client2 == mock_client_instance
        assert mock_openai_class.call_count == 1  # Not called again


class TestIntegration:
    """Integration tests for the full flow."""

    def test_full_flow_with_english(self):
        """Full flow: English user gets English message."""
        state = {
            "user_response_language": "en",
            "query_language": "en",
        }
        result = status_messages.get_status_message(
            status_messages.PROCESSING_ERROR, state
        )
        assert (
            result
            == "It looks like I'm having trouble processing your message. Please report this issue to my creators."
        )

    def test_full_flow_with_query_language_fallback(self):
        """Full flow: No user preference, uses query language."""
        state = {
            "user_response_language": None,
            "query_language": "en",
        }
        result = status_messages.get_status_message(
            status_messages.FINALIZING_RESPONSE, state
        )
        assert (
            result
            == "I'm pulling everything together into a helpful response for you."
        )

    @patch("bt_servant_engine.services.status_messages._translate_dynamically")
    def test_full_flow_with_dynamic_translation(self, mock_translate):
        """Full flow: Unsupported language triggers dynamic translation."""
        mock_translate.return_value = "Mensaje en italiano"
        state = {
            "user_response_language": "it",  # Italian not pre-loaded
            "query_language": LANGUAGE_UNKNOWN,
        }
        result = status_messages.get_status_message(
            status_messages.PACKAGING_VOICE_RESPONSE, state
        )
        assert result == "Mensaje en italiano"
        mock_translate.assert_called_once_with(
            status_messages.PACKAGING_VOICE_RESPONSE, "it"
        )
