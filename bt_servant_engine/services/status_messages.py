"""Status message localization for user-facing progress and error messages.

This module provides infrastructure for delivering status messages in the user's
preferred language, with pre-loaded translations for supported languages and
dynamic translation fallback for other languages.

Dynamic translations are persisted back to the JSON file in DATA_DIR, allowing
the system to "learn" new languages over time and avoid redundant API calls.
The file persists across container restarts when DATA_DIR is mounted as a volume.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional, TypedDict, cast

from openai import OpenAI, OpenAIError

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

# Template file location (bundled with package)
_TEMPLATE_PATH = Path(__file__).parent / "status_messages_template.json"

# Runtime data file location (in DATA_DIR for persistence)
DATA_DIR = Path(config.DATA_DIR)
DATA_DIR.mkdir(parents=True, exist_ok=True)
_STATUS_MESSAGES_PATH = DATA_DIR / "status_messages_data.json"

# Default emoji overrides applied unless superseded by configuration
_DEFAULT_PROGRESS_MESSAGE_EMOJI_OVERRIDES: Dict[str, str] = {
    THINKING_ABOUT_MESSAGE: "🧠",
    FOUND_RELEVANT_DOCUMENTS: "📚",
}


@dataclass(slots=True)
class StatusMessageStore:
    """In-memory view of status message translations and dynamic cache."""

    status_messages: dict[str, dict[str, str]]
    dynamic_cache: dict[tuple[str, str], str]

    def cache_translation(self, message_key: str, language: str, text: str) -> None:
        """Add a dynamic translation to the in-memory cache."""
        self.dynamic_cache[(message_key, language)] = text


class LocalizedProgressMessage(TypedDict):
    """Structured progress status message with emoji metadata."""

    key: str
    text: str
    emoji: str


def _initialize_status_messages_file() -> None:
    """Initialize status messages file in DATA_DIR if it doesn't exist.

    On first startup, copies the template from the package to DATA_DIR.
    This allows the file to be writable and persist across container restarts.
    """
    if not _STATUS_MESSAGES_PATH.exists():
        logger.info(
            "Status messages file not found in DATA_DIR, copying template from package: %s -> %s",
            _TEMPLATE_PATH,
            _STATUS_MESSAGES_PATH,
        )
        shutil.copy2(_TEMPLATE_PATH, _STATUS_MESSAGES_PATH)
        logger.info("Status messages template copied successfully")


# Initialize the file on module load
_initialize_status_messages_file()


def _load_status_messages() -> dict[str, dict[str, str]]:
    with open(_STATUS_MESSAGES_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return cast(dict[str, dict[str, str]], data)


_STATUS_STORE = StatusMessageStore(
    status_messages=_load_status_messages(),
    dynamic_cache={},
)

# Expose dynamic cache for testability without encouraging direct mutation.
_DYNAMIC_TRANSLATION_CACHE = _STATUS_STORE.dynamic_cache


def get_dynamic_translation_cache() -> dict[tuple[str, str], str]:
    """Return the in-memory dynamic translation cache."""
    return _STATUS_STORE.dynamic_cache


def clear_dynamic_translation_cache() -> None:
    """Clear cached dynamic translations (useful in tests)."""
    _STATUS_STORE.dynamic_cache.clear()


def update_dynamic_translation_cache(entries: dict[tuple[str, str], str]) -> None:
    """Populate the dynamic cache with the provided translations."""
    _STATUS_STORE.dynamic_cache.update(entries)


@lru_cache(maxsize=1)
def _get_openai_client() -> OpenAI:
    """Lazy initialize OpenAI client."""
    return OpenAI(api_key=config.OPENAI_API_KEY)


def reset_openai_client_cache() -> None:
    """Reset the cached OpenAI client (primarily for tests)."""
    _get_openai_client.cache_clear()


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
            data = cast(dict[str, dict[str, str]], json.load(json_file))

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

            _STATUS_STORE.status_messages = data

        except (OSError, ValueError, TypeError):
            # Clean up temp file if something went wrong
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            raise

    except (OSError, json.JSONDecodeError, ValueError):
        logger.exception(
            "Failed to persist translation for message '%s' in language '%s'",
            message_key,
            language,
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
    cached = _STATUS_STORE.dynamic_cache.get(cache_key)
    if cached is not None:
        return cached

    # Get the English version as source
    english_text = _STATUS_STORE.status_messages.get(message_key, {}).get("en", "")
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
        _STATUS_STORE.cache_translation(message_key, target_language, translated_text)

        # Persist the translation to JSON file for future use
        _persist_translation(message_key, target_language, translated_text)

        logger.info(
            "Dynamically translated status message '%s' to language '%s'",
            message_key,
            target_language,
        )
        return translated_text

    except (OpenAIError, ValueError, TypeError, RuntimeError):
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
        **format_params: Optional format parameters for template messages
            (e.g., action="do something")

    Returns:
        Localized status message text, with format parameters applied if provided
    """
    target_language = get_effective_response_language(state)

    # Check if message key exists
    if message_key not in _STATUS_STORE.status_messages:
        logger.error("Unknown status message key: %s", message_key)
        return f"[Unknown message: {message_key}]"

    # Check for pre-loaded translation
    translations = _STATUS_STORE.status_messages[message_key]
    if target_language in translations:
        message = translations[target_language]
    else:
        # Fall back to dynamic translation
        logger.debug(
            (
                "No pre-loaded translation for message '%s' in language '%s'; "
                "using dynamic translation"
            ),
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


def _resolve_progress_emoji(message_key: str) -> str:
    """Select the emoji to display for a status message.

    Priority:
        1. Explicit override from configuration (PROGRESS_MESSAGE_EMOJI_OVERRIDES)
        2. Built-in defaults defined in this module
        3. Global default emoji from configuration (PROGRESS_MESSAGE_EMOJI)
    """
    overrides_value = getattr(config, "PROGRESS_MESSAGE_EMOJI_OVERRIDES", {})
    if isinstance(overrides_value, dict):
        maybe_override = overrides_value.get(message_key)
        if isinstance(maybe_override, str):
            return maybe_override
    if message_key in _DEFAULT_PROGRESS_MESSAGE_EMOJI_OVERRIDES:
        return _DEFAULT_PROGRESS_MESSAGE_EMOJI_OVERRIDES[message_key]
    return config.PROGRESS_MESSAGE_EMOJI


def get_progress_message(
    message_key: str, state: Any, **format_params: Any
) -> LocalizedProgressMessage:
    """Return localized progress message text and emoji metadata."""
    text = get_status_message(message_key, state, **format_params)
    emoji = _resolve_progress_emoji(message_key)
    return cast(
        LocalizedProgressMessage,
        {"key": message_key, "text": text, "emoji": emoji},
    )


def make_progress_message(
    text: str, *, message_key: str = "", emoji: Optional[str] = None
) -> LocalizedProgressMessage:
    """Build a custom progress message with optional emoji override."""
    final_emoji = emoji if emoji is not None else config.PROGRESS_MESSAGE_EMOJI
    return cast(
        LocalizedProgressMessage,
        {"key": message_key, "text": text, "emoji": final_emoji},
    )


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
    # Functions
    "get_status_message",
    "get_effective_response_language",
    "get_progress_message",
    "get_dynamic_translation_cache",
    "clear_dynamic_translation_cache",
    "update_dynamic_translation_cache",
    "reset_openai_client_cache",
    "LocalizedProgressMessage",
    "make_progress_message",
]
