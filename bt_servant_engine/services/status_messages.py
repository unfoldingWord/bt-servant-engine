"""Status message localization for user-facing progress and error messages.

This module provides infrastructure for delivering status messages in the user's
preferred language, with pre-loaded translations for supported languages and
dynamic translation fallback for other languages.

Dynamic translations are persisted back to the JSON file, allowing the system
to "learn" new languages over time and avoid redundant API calls.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Optional

from openai import OpenAI

from bt_servant_engine.core.config import config
from bt_servant_engine.core.language import LANGUAGE_UNKNOWN
from bt_servant_engine.core.logging import get_logger

logger = get_logger(__name__)

# Message key constants
THINKING_ABOUT_MESSAGE = "THINKING_ABOUT_MESSAGE"
SEARCHING_BIBLE_RESOURCES = "SEARCHING_BIBLE_RESOURCES"
REVIEWING_FIA_GUIDANCE = "REVIEWING_FIA_GUIDANCE"
CHECKING_CAPABILITIES = "CHECKING_CAPABILITIES"
GENERATING_HELP_RESPONSE = "GENERATING_HELP_RESPONSE"
GATHERING_PASSAGE_SUMMARY = "GATHERING_PASSAGE_SUMMARY"
EXTRACTING_KEYWORDS = "EXTRACTING_KEYWORDS"
COMPILING_TRANSLATION_HELPS = "COMPILING_TRANSLATION_HELPS"
GATHERING_PASSAGE_TEXT = "GATHERING_PASSAGE_TEXT"
PREPARING_AUDIO = "PREPARING_AUDIO"
TRANSLATING_PASSAGE = "TRANSLATING_PASSAGE"
TRANSLATING_RESPONSE = "TRANSLATING_RESPONSE"
FINALIZING_RESPONSE = "FINALIZING_RESPONSE"
TRANSCRIBING_VOICE = "TRANSCRIBING_VOICE"
PACKAGING_VOICE_RESPONSE = "PACKAGING_VOICE_RESPONSE"
PROCESSING_ERROR = "PROCESSING_ERROR"
FOUND_RELEVANT_DOCUMENTS = "FOUND_RELEVANT_DOCUMENTS"
CONTINUATION_PROMPT_TEMPLATE = "CONTINUATION_PROMPT_TEMPLATE"

# Load pre-translated messages
_STATUS_MESSAGES_PATH = Path(__file__).parent / "status_messages_data.json"
with open(_STATUS_MESSAGES_PATH, encoding="utf-8") as f:
    _STATUS_MESSAGES: dict[str, dict[str, str]] = json.load(f)

# Cache for dynamically translated messages: (message_key, language) -> translated_text
_DYNAMIC_TRANSLATION_CACHE: dict[tuple[str, str], str] = {}

# OpenAI client for dynamic translations
_openai_client: Optional[OpenAI] = None


def _get_openai_client() -> OpenAI:
    """Lazy initialize OpenAI client."""
    global _openai_client  # pylint: disable=global-statement
    if _openai_client is None:
        _openai_client = OpenAI(api_key=config.OPENAI_API_KEY)
    return _openai_client


def _persist_translation(message_key: str, language: str, translation: str) -> None:
    """Persist a dynamic translation back to the JSON file.

    Uses atomic write operation (write to temp file, then rename) to avoid
    corrupting the file if the process crashes mid-write.

    Args:
        message_key: The message key
        language: ISO 639-1 language code
        translation: The translated text
    """
    try:
        # Read current data
        with open(_STATUS_MESSAGES_PATH, encoding="utf-8") as json_file:
            data = json.load(json_file)

        # Add the new translation
        if message_key in data:
            data[message_key][language] = translation
        else:
            logger.warning("Cannot persist translation for unknown message key: %s", message_key)
            return

        # Write atomically using temp file + rename
        temp_fd, temp_path = tempfile.mkstemp(
            dir=_STATUS_MESSAGES_PATH.parent, prefix=".status_messages_", suffix=".json.tmp"
        )
        try:
            with os.fdopen(temp_fd, "w", encoding="utf-8") as temp_file:
                json.dump(data, temp_file, indent=2, ensure_ascii=False)
                temp_file.write("\n")  # Add trailing newline

            # Atomic rename
            os.replace(temp_path, _STATUS_MESSAGES_PATH)
            logger.info(
                "Persisted dynamic translation for message '%s' in language '%s'",
                message_key,
                language,
            )

            # Update in-memory cache
            global _STATUS_MESSAGES  # pylint: disable=global-statement
            _STATUS_MESSAGES = data

        except Exception:  # pylint: disable=broad-exception-caught
            # Clean up temp file if something went wrong
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            raise

    except Exception:  # pylint: disable=broad-exception-caught
        logger.error(
            "Failed to persist translation for message '%s' in language '%s'",
            message_key,
            language,
            exc_info=True,
        )
        # Don't crash the application if persistence fails


def get_effective_response_language(state: Any) -> str:
    """Determine the effective language for responses and status messages.

    This uses the same priority logic as the response translation pipeline:
    1. User's set preference (user_response_language)
    2. Detected query language (query_language)
    3. English fallback ("en")

    Args:
        state: The BrainState dictionary containing language information

    Returns:
        ISO 639-1 language code (e.g., "en", "ar", "fr")
    """
    # Priority 1: User's explicit preference
    user_language = state.get("user_response_language")
    if user_language:
        return str(user_language).strip().lower()

    # Priority 2: Detected query language
    query_language = state.get("query_language")
    if query_language and query_language != LANGUAGE_UNKNOWN:
        return str(query_language).strip().lower()

    # Priority 3: Default to English
    return "en"


def _translate_dynamically(message_key: str, target_language: str) -> str:
    """Use OpenAI to translate a status message to the target language.

    Args:
        message_key: The message key to translate
        target_language: ISO 639-1 language code

    Returns:
        Translated message text
    """
    cache_key = (message_key, target_language)
    if cache_key in _DYNAMIC_TRANSLATION_CACHE:
        return _DYNAMIC_TRANSLATION_CACHE[cache_key]

    # Get the English version as source
    english_text = _STATUS_MESSAGES.get(message_key, {}).get("en", "")
    if not english_text:
        logger.warning("No English text found for message key: %s", message_key)
        return english_text

    try:
        client = _get_openai_client()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a translator. Translate the following status message "
                        f"into the language with ISO 639-1 code '{target_language}'. "
                        "Preserve the tone and meaning. Return only the translated text."
                    ),
                },
                {"role": "user", "content": english_text},
            ],
            temperature=0.3,
        )
        translated_text = response.choices[0].message.content or english_text
        translated_text = translated_text.strip()

        # Cache the translation in memory
        _DYNAMIC_TRANSLATION_CACHE[cache_key] = translated_text

        # Persist the translation to JSON file for future use
        _persist_translation(message_key, target_language, translated_text)

        logger.info(
            "Dynamically translated status message '%s' to language '%s'",
            message_key,
            target_language,
        )
        return translated_text

    except Exception:  # pylint: disable=broad-exception-caught
        logger.warning(
            "Failed to dynamically translate message '%s' to '%s', using English fallback",
            message_key,
            target_language,
            exc_info=True,
        )
        return english_text


def get_status_message(message_key: str, state: Any, **format_params: Any) -> str:
    """Get a localized status message for the given key and state.

    Args:
        message_key: One of the message key constants (e.g., THINKING_ABOUT_MESSAGE)
        state: The BrainState dictionary containing language information
        **format_params: Optional format parameters for template messages (e.g., action="do something")

    Returns:
        Localized status message text, with format parameters applied if provided
    """
    target_language = get_effective_response_language(state)

    # Check if message key exists
    if message_key not in _STATUS_MESSAGES:
        logger.error("Unknown status message key: %s", message_key)
        return f"[Unknown message: {message_key}]"

    # Check for pre-loaded translation
    translations = _STATUS_MESSAGES[message_key]
    if target_language in translations:
        message = translations[target_language]
    else:
        # Fall back to dynamic translation
        logger.debug(
            "No pre-loaded translation for message '%s' in language '%s', using dynamic translation",
            message_key,
            target_language,
        )
        message = _translate_dynamically(message_key, target_language)

    # Apply format parameters if provided
    if format_params:
        try:
            message = message.format(**format_params)
        except KeyError as e:
            logger.warning(
                "Format parameter %s not found in message '%s': %s", e, message_key, message
            )

    return message


__all__ = [
    # Message key constants
    "THINKING_ABOUT_MESSAGE",
    "SEARCHING_BIBLE_RESOURCES",
    "REVIEWING_FIA_GUIDANCE",
    "CHECKING_CAPABILITIES",
    "GENERATING_HELP_RESPONSE",
    "GATHERING_PASSAGE_SUMMARY",
    "EXTRACTING_KEYWORDS",
    "COMPILING_TRANSLATION_HELPS",
    "GATHERING_PASSAGE_TEXT",
    "PREPARING_AUDIO",
    "TRANSLATING_PASSAGE",
    "TRANSLATING_RESPONSE",
    "FINALIZING_RESPONSE",
    "TRANSCRIBING_VOICE",
    "PACKAGING_VOICE_RESPONSE",
    "PROCESSING_ERROR",
    "FOUND_RELEVANT_DOCUMENTS",
    "CONTINUATION_PROMPT_TEMPLATE",
    # Functions
    "get_status_message",
    "get_effective_response_language",
]
