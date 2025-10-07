"""Intent types and models for the BT Servant application."""

from enum import Enum

from pydantic import BaseModel


class IntentType(str, Enum):
    """Enumeration of all supported user intents in the graph."""

    GET_BIBLE_TRANSLATION_ASSISTANCE = "get-bible-translation-assistance"
    CONSULT_FIA_RESOURCES = "consult-fia-resources"
    GET_PASSAGE_SUMMARY = "get-passage-summary"
    GET_PASSAGE_KEYWORDS = "get-passage-keywords"
    GET_TRANSLATION_HELPS = "get-translation-helps"
    RETRIEVE_SCRIPTURE = "retrieve-scripture"
    LISTEN_TO_SCRIPTURE = "listen-to-scripture"
    TRANSLATE_SCRIPTURE = "translate-scripture"
    PERFORM_UNSUPPORTED_FUNCTION = "perform-unsupported-function"
    RETRIEVE_SYSTEM_INFORMATION = "retrieve-system-information"
    SET_RESPONSE_LANGUAGE = "set-response-language"
    SET_AGENTIC_STRENGTH = "set-agentic-strength"
    CONVERSE_WITH_BT_SERVANT = "converse-with-bt-servant"


class UserIntents(BaseModel):
    """Container for a list of user intents."""

    intents: list[IntentType]


__all__ = ["IntentType", "UserIntents"]
