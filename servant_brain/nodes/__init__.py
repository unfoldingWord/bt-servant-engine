"""LangGraph node implementations extracted from the legacy brain module."""
# pylint: disable=duplicate-code

from .capabilities import (
    capabilities,
    build_boilerplate_message,
    build_full_help_message,
    BOILER_PLATE_AVAILABLE_FEATURES_MESSAGE,
    FULL_HELP_MESSAGE,
    FIRST_INTERACTION_MESSAGE,
    CONVERSE_AGENT_SYSTEM_PROMPT,
    HELP_AGENT_SYSTEM_PROMPT,
    UNSUPPORTED_FUNCTION_AGENT_SYSTEM_PROMPT,
)
from .conversation import (
    start,
    handle_unsupported_function,
    handle_system_information_request,
    converse_with_bt_servant,
)
from .classification import determine_intents
from .language import detect_language, determine_query_language, set_response_language
from .preprocess import (
    PREPROCESSOR_AGENT_SYSTEM_PROMPT,
    PreprocessorResult,
    preprocess_user_query,
)
from .retrieval import query_vector_db, query_open_ai
from .responses import (
    combine_responses,
    translate_text,
    translate_responses,
    chunk_message,
    needs_chunking,
)

__all__ = [
    "capabilities",
    "build_boilerplate_message",
    "build_full_help_message",
    "BOILER_PLATE_AVAILABLE_FEATURES_MESSAGE",
    "FULL_HELP_MESSAGE",
    "FIRST_INTERACTION_MESSAGE",
    "CONVERSE_AGENT_SYSTEM_PROMPT",
    "HELP_AGENT_SYSTEM_PROMPT",
    "UNSUPPORTED_FUNCTION_AGENT_SYSTEM_PROMPT",
    "start",
    "handle_unsupported_function",
    "handle_system_information_request",
    "converse_with_bt_servant",
    "determine_intents",
    "detect_language",
    "determine_query_language",
    "set_response_language",
    "PREPROCESSOR_AGENT_SYSTEM_PROMPT",
    "PreprocessorResult",
    "preprocess_user_query",
    "query_vector_db",
    "query_open_ai",
    "combine_responses",
    "translate_text",
    "translate_responses",
    "chunk_message",
    "needs_chunking",
]
