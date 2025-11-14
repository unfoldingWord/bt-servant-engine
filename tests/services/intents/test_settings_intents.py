"""Tests for settings-related intents (response language, agentic strength)."""

from bt_servant_engine.core.intents import IntentType
from bt_servant_engine.services.intents.settings_intents import (
    ClearResponseLanguageDependencies,
    ClearResponseLanguageRequest,
    clear_response_language,
)


def test_clear_response_language_resets_preference():
    """Clearing response language removes the stored preference and responds."""
    calls: list[str] = []

    def fake_clear(user_id: str) -> None:
        calls.append(user_id)

    request = ClearResponseLanguageRequest(user_id="user-42")
    dependencies = ClearResponseLanguageDependencies(
        clear_user_response_language=fake_clear
    )

    result = clear_response_language(request, dependencies)

    assert calls == ["user-42"]
    assert result["user_response_language"] is None
    responses = result["responses"]
    assert responses and responses[0]["intent"] == IntentType.CLEAR_RESPONSE_LANGUAGE
    assert "Cleared your response-language preference" in responses[0]["response"]
