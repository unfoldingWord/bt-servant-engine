"""Status message localization for user-facing progress and error messages.

This module provides infrastructure for delivering status messages in the user's
preferred language, with pre-loaded translations for supported languages and
dynamic translation fallback for other languages.

Dynamic translations are persisted back to the JSON file in DATA_DIR, allowing
the system to "learn" new languages over time and avoid redundant API calls.
The file persists across container restarts when DATA_DIR is mounted as a volume.
"""

from __future__ import annotations

import copy
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional, TypedDict, cast

from openai import OpenAI, OpenAIError

from bt_servant_engine.core.config import config
from bt_servant_engine.core.language import LANGUAGE_UNKNOWN
from bt_servant_engine.core.logging import get_logger

logger = get_logger(__name__)

# Message key constants
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

# ReAct Agentic MCP message keys
REACT_CONNECTING_TO_RESOURCES = "REACT_CONNECTING_TO_RESOURCES"
REACT_ANALYZING_REQUEST = "REACT_ANALYZING_REQUEST"
REACT_FETCHING_SCRIPTURE = "REACT_FETCHING_SCRIPTURE"
REACT_FETCHING_NOTES = "REACT_FETCHING_NOTES"
REACT_FETCHING_WORD_INFO = "REACT_FETCHING_WORD_INFO"
REACT_DISCOVERING_RESOURCES = "REACT_DISCOVERING_RESOURCES"
REACT_COMBINING_RESULTS = "REACT_COMBINING_RESULTS"
REACT_LANGUAGE_FALLBACK = "REACT_LANGUAGE_FALLBACK"

# Template file location (bundled with package)
_TEMPLATE_PATH = Path(__file__).parent / "status_messages_template.json"

# Runtime data file location (in DATA_DIR for persistence)
DATA_DIR = Path(config.DATA_DIR)
DATA_DIR.mkdir(parents=True, exist_ok=True)
_STATUS_MESSAGES_PATH = DATA_DIR / "status_messages_data.json"


@dataclass(slots=True)
class StatusMessageStore:
    """In-memory view of status message translations and dynamic cache."""

    status_messages: dict[str, dict[str, str]]
    dynamic_cache: dict[tuple[str, str], str]

    def cache_translation(self, message_key: str, language: str, text: str) -> None:
        """Add a dynamic translation to the in-memory cache."""
        self.dynamic_cache[(message_key, language)] = text


class LocalizedProgressMessage(TypedDict):
    """Structured progress status message (emoji field retained for compatibility)."""

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

    def _sanitize_translation(translated: str) -> str:
        """Clamp pathological translations (e.g., runaway repetition)."""
        translated = translated.strip()
        if not translated:
            return english_text

        max_len = 500
        ratio_limit = 8
        if len(translated) > max(max_len, len(english_text) * ratio_limit):
            logger.warning(
                "Discarding oversized translation for message '%s' language '%s' "
                "(len=%d, english_len=%d): %s",
                message_key,
                target_language,
                len(translated),
                len(english_text),
                translated[:200],
            )
            return english_text
        return translated

    try:
        client = _get_openai_client()
        response = client.chat.completions.create(
            model="gpt-4o",
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
        translated_text = _sanitize_translation(translated_text)

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
        message_key: One of the message key constants (e.g., REVIEWING_FIA_GUIDANCE)
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
        return _apply_whatsapp_italics(f"[Unknown message: {message_key}]")

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

    return _apply_whatsapp_italics(message)


def _apply_whatsapp_italics(message: str) -> str:
    """Wrap a message in underscores so it renders as italics."""
    sanitized = message.strip()
    if not sanitized:
        return ""
    if sanitized.startswith("_") and sanitized.endswith("_"):
        return sanitized
    return f"_{sanitized}_"


def _resolve_progress_emoji(_: str) -> str:
    """Return an empty string to disable emoji usage in status messages."""
    return ""


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
    text: str, *, message_key: str = "", _emoji: Optional[str] = None
) -> LocalizedProgressMessage:
    """Build a custom progress message (emojis disabled)."""
    italicized = _apply_whatsapp_italics(text)
    return cast(
        LocalizedProgressMessage,
        {"key": message_key, "text": italicized, "emoji": ""},
    )


def list_status_message_cache() -> dict[str, dict[str, str]]:
    """Return a deep copy of the current status message cache."""
    return copy.deepcopy(_STATUS_STORE.status_messages)


def get_status_message_translations(message_key: str) -> dict[str, str]:
    """Return translations for a specific message key."""
    translations = _STATUS_STORE.status_messages.get(message_key)
    if translations is None:
        raise KeyError(message_key)
    return copy.deepcopy(translations)


def list_status_messages_for_language(language: str) -> dict[str, str]:
    """Return all message texts for a given language keyed by message id."""
    normalized = language.strip().lower()
    results: dict[str, str] = {}
    for message_key, translations in _STATUS_STORE.status_messages.items():
        text = translations.get(normalized)
        if text:
            results[message_key] = text
    return results


def _persist_status_messages(messages: dict[str, dict[str, str]]) -> None:
    """Persist the full status message map to disk atomically."""
    temp_fd, temp_path = tempfile.mkstemp(
        dir=_STATUS_MESSAGES_PATH.parent, prefix=".status_messages_", suffix=".json.tmp"
    )
    try:
        with os.fdopen(temp_fd, "w", encoding="utf-8") as temp_file:
            json.dump(messages, temp_file, indent=2, ensure_ascii=False)
            temp_file.write("\n")
        os.replace(temp_path, _STATUS_MESSAGES_PATH)
    finally:
        if os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except OSError:
                pass


def set_status_message_translation(message_key: str, language: str, text: str) -> None:
    """Create or update a translation for the given message key and language."""
    if not text.strip():
        raise ValueError("translation text cannot be empty")
    if language.lower().strip() == "en":
        raise ValueError("cannot override the English source translation")
    messages = _STATUS_STORE.status_messages.get(message_key)
    if messages is None:
        raise KeyError(message_key)

    updated_messages = copy.deepcopy(_STATUS_STORE.status_messages)
    updated_messages.setdefault(message_key, {})[language] = text

    _persist_status_messages(updated_messages)
    _STATUS_STORE.status_messages = updated_messages
    _STATUS_STORE.cache_translation(message_key, language, text)


def delete_status_message_translation(message_key: str, language: str) -> None:
    """Delete a translation for the given message key and language."""
    if language.lower().strip() == "en":
        raise ValueError("cannot delete the English source translation")
    messages = _STATUS_STORE.status_messages.get(message_key)
    if messages is None:
        raise KeyError(message_key)
    if language not in messages:
        raise KeyError(f"{message_key}:{language}")

    updated_messages = copy.deepcopy(_STATUS_STORE.status_messages)
    updated_messages[message_key].pop(language, None)
    _persist_status_messages(updated_messages)
    _STATUS_STORE.status_messages = updated_messages
    _STATUS_STORE.dynamic_cache.pop((message_key, language), None)


def delete_status_messages_for_language(language: str) -> None:
    """Delete all translations for a given language across all message keys."""
    normalized = language.lower().strip()
    if normalized == "en":
        raise ValueError("cannot delete the English source translation")

    updated_messages = copy.deepcopy(_STATUS_STORE.status_messages)
    removed_any = False
    for key, translations in updated_messages.items():
        if normalized in translations:
            translations.pop(normalized, None)
            _STATUS_STORE.dynamic_cache.pop((key, normalized), None)
            removed_any = True

    if not removed_any:
        raise KeyError(normalized)

    _persist_status_messages(updated_messages)
    _STATUS_STORE.status_messages = updated_messages


__all__ = [
    # Message key constants
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
    # ReAct Agentic MCP message keys
    "REACT_CONNECTING_TO_RESOURCES",
    "REACT_ANALYZING_REQUEST",
    "REACT_FETCHING_SCRIPTURE",
    "REACT_FETCHING_NOTES",
    "REACT_FETCHING_WORD_INFO",
    "REACT_DISCOVERING_RESOURCES",
    "REACT_COMBINING_RESULTS",
    "REACT_LANGUAGE_FALLBACK",
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
    "list_status_message_cache",
    "get_status_message_translations",
    "list_status_messages_for_language",
    "set_status_message_translation",
    "delete_status_message_translation",
    "delete_status_messages_for_language",
]
