"""Decision graph and message-processing pipeline for BT Servant.

This module now serves primarily as a façade that re-exports the LangGraph
nodes and helper types from the `servant_brain` package. The legacy API
remains for compatibility while the underlying implementations live in
smaller, focused modules.
"""

from __future__ import annotations

from collections.abc import Hashable
from typing import Any, List

import servant_brain.selection as selection_helpers
from servant_brain import dependencies as _deps
from servant_brain.graph import (
    create_brain as _graph_create,
    process_intents as _graph_process_intents,
)
from servant_brain.classifier import IntentType, UserIntents
from servant_brain.intents import listen_to_scripture as _intent_listen_to_scripture
from servant_brain.intents import passage_keywords as _intent_passage_keywords
from servant_brain.intents import passage_summary as _intent_passage_summary
from servant_brain.intents import retrieve_scripture as _intent_retrieve_scripture
from servant_brain.intents import translate_responses as _intent_translate_responses
from servant_brain.intents import translate_scripture as _intent_translate_scripture
from servant_brain.intents import translation_helps as _intent_translation_helps
from servant_brain.models import TranslatedPassage
from servant_brain.nodes.capabilities import (
    capabilities,
    build_boilerplate_message,
    build_full_help_message,
    BOILER_PLATE_AVAILABLE_FEATURES_MESSAGE,
    CONVERSE_AGENT_SYSTEM_PROMPT,
    FIRST_INTERACTION_MESSAGE,
    FULL_HELP_MESSAGE,
    HELP_AGENT_SYSTEM_PROMPT,
    UNSUPPORTED_FUNCTION_AGENT_SYSTEM_PROMPT,
)
from servant_brain.nodes.classification import determine_intents
from servant_brain.nodes.conversation import (
    converse_with_bt_servant,
    handle_system_information_request,
    handle_unsupported_function,
    start,
)
from servant_brain.nodes.language import (
    detect_language,
    determine_query_language,
    set_response_language,
)
from servant_brain.nodes.preprocess import (
    PREPROCESSOR_AGENT_SYSTEM_PROMPT,
    PreprocessorResult,
    preprocess_user_query,
)
from servant_brain.nodes import responses as _responses_nodes
from servant_brain.nodes.responses import (
    chunk_message,
    combine_responses,
    needs_chunking,
    translate_text,
)
from servant_brain.nodes.retrieval import query_open_ai, query_vector_db
from servant_brain.prompts import (
    CHOP_AGENT_SYSTEM_PROMPT as _CHOP_AGENT_SYSTEM_PROMPT,
    COMBINE_RESPONSES_SYSTEM_PROMPT as _COMBINE_RESPONSES_SYSTEM_PROMPT,
    FINAL_RESPONSE_AGENT_SYSTEM_PROMPT as _FINAL_RESPONSE_AGENT_SYSTEM_PROMPT,
    PASSAGE_SELECTION_AGENT_SYSTEM_PROMPT as _PASSAGE_SELECTION_AGENT_SYSTEM_PROMPT,
    RESPONSE_TRANSLATOR_SYSTEM_PROMPT as _RESPONSE_TRANSLATOR_SYSTEM_PROMPT,
    TRANSLATE_PASSAGE_AGENT_SYSTEM_PROMPT as _TRANSLATE_PASSAGE_AGENT_SYSTEM_PROMPT,
)
from servant_brain.state import BrainState, Capability
from servant_brain.tokens import extract_cached_input_tokens as _extract_cached_input_tokens
from servant_brain.language import (
    DETECT_LANGUAGE_AGENT_SYSTEM_PROMPT,
    Language,
    MessageLanguage,
    ResponseLanguage,
    SET_RESPONSE_LANGUAGE_AGENT_SYSTEM_PROMPT,
    TARGET_TRANSLATION_LANGUAGE_AGENT_SYSTEM_PROMPT,
)
from utils.bsb import BOOK_MAP as BSB_BOOK_MAP, label_ranges, normalize_book_name, select_verses
from utils.bible_data import list_available_sources, load_book_titles, resolve_bible_data_root
from utils.bible_locale import get_book_name
from utils.perf import add_tokens
from config import config as _config

open_ai_client = _deps.open_ai_client
supported_language_map = _deps.supported_language_map
LANGUAGE_UNKNOWN = _deps.LANGUAGE_UNKNOWN
RELEVANCE_CUTOFF = _deps.RELEVANCE_CUTOFF
TOP_K = _deps.TOP_K

CHOP_AGENT_SYSTEM_PROMPT = _CHOP_AGENT_SYSTEM_PROMPT
COMBINE_RESPONSES_SYSTEM_PROMPT = _COMBINE_RESPONSES_SYSTEM_PROMPT
FINAL_RESPONSE_AGENT_SYSTEM_PROMPT = _FINAL_RESPONSE_AGENT_SYSTEM_PROMPT
PASSAGE_SELECTION_AGENT_SYSTEM_PROMPT = _PASSAGE_SELECTION_AGENT_SYSTEM_PROMPT
RESPONSE_TRANSLATOR_SYSTEM_PROMPT = _RESPONSE_TRANSLATOR_SYSTEM_PROMPT
TRANSLATE_PASSAGE_AGENT_SYSTEM_PROMPT = _TRANSLATE_PASSAGE_AGENT_SYSTEM_PROMPT
config = _config
PassageSelection = selection_helpers.PassageSelection
PassageRef = selection_helpers.PassageRef


def process_intents(state: Any) -> List[Hashable]:  # pylint: disable=too-many-branches
    """Delegate to the split graph helper to decide next nodes."""

    return _graph_process_intents(state)


def _resolve_selection_for_single_book(
    query: str,
    query_lang: str,
) -> tuple[str | None, list[tuple[int, int | None, int | None, int | None]] | None, str | None]:
    """Thin wrapper delegating to servant_brain.selection with the same contract."""

    return selection_helpers.resolve_selection_for_single_book(
        query=query,
        query_lang=query_lang,
        open_ai_client=open_ai_client,
        passage_selection_model=selection_helpers.PassageSelection,
        passage_ref_model=selection_helpers.PassageRef,
        add_tokens=add_tokens,
        extract_cached_input_tokens=_extract_cached_input_tokens,
        translate_text=translate_text,
        books_map=BSB_BOOK_MAP,
        normalize_book_name=normalize_book_name,
        selection_prompt_template=PASSAGE_SELECTION_AGENT_SYSTEM_PROMPT,
    )


def translate_responses(state: Any) -> dict:
    """Wrapper ensuring detect_language patches against brain module apply."""

    # pylint: disable=protected-access
    return _intent_translate_responses.translate_responses(
        state,
        combine_responses=combine_responses,
        is_protected_response_item=(
            _responses_nodes._is_protected_response_item  # type: ignore[attr-defined]
        ),
        reconstruct_structured_text=(
            _responses_nodes._reconstruct_structured_text  # type: ignore[attr-defined]
        ),
        detect_language=detect_language,
        translate_text=translate_text,
        supported_language_map=supported_language_map,
    )


def get_passage_summary(state: Any) -> dict:
    """Thin wrapper delegating to servant_brain.intents.passage_summary."""

    return _intent_passage_summary.get_passage_summary(
        state,
        resolve_selection_for_single_book=_resolve_selection_for_single_book,
        open_ai_client=open_ai_client,
        add_tokens=add_tokens,
        extract_cached_input_tokens=_extract_cached_input_tokens,
    )


def get_passage_keywords(state: Any) -> dict:
    """Thin wrapper delegating to servant_brain.intents.passage_keywords."""

    return _intent_passage_keywords.get_passage_keywords(
        state,
        resolve_selection_for_single_book=_resolve_selection_for_single_book,
    )


def get_translation_helps(state: Any) -> dict:
    """Thin wrapper delegating to servant_brain.intents.translation_helps."""

    return _intent_translation_helps.get_translation_helps(
        state,
        resolve_selection_for_single_book=_resolve_selection_for_single_book,
        open_ai_client=open_ai_client,
        add_tokens=add_tokens,
        extract_cached_input_tokens=_extract_cached_input_tokens,
    )


def retrieve_scripture(state: Any) -> dict:  # pylint: disable=too-many-arguments
    """Thin wrapper delegating to servant_brain.intents.retrieve_scripture."""

    return _intent_retrieve_scripture.retrieve_scripture(
        state,
        resolve_selection_for_single_book=_resolve_selection_for_single_book,
        resolve_bible_data_root=resolve_bible_data_root,
        list_available_sources=list_available_sources,
        load_book_titles=load_book_titles,
        get_book_name=get_book_name,
        select_verses=select_verses,
        label_ranges=label_ranges,
        translate_text=translate_text,
        open_ai_client=open_ai_client,
        add_tokens=add_tokens,
        extract_cached_input_tokens=_extract_cached_input_tokens,
        ResponseLanguage=ResponseLanguage,
        Language=Language,
        TARGET_TRANSLATION_LANGUAGE_AGENT_SYSTEM_PROMPT=(
            TARGET_TRANSLATION_LANGUAGE_AGENT_SYSTEM_PROMPT
        ),
        supported_language_map=supported_language_map,
    )


def listen_to_scripture(state: Any) -> dict:
    """Thin wrapper delegating to servant_brain.intents.listen_to_scripture."""

    return _intent_listen_to_scripture.listen_to_scripture(state, retrieve=retrieve_scripture)


def translate_scripture(state: Any) -> dict:
    """Thin wrapper delegating to servant_brain.intents.translate_scripture."""

    return _intent_translate_scripture.translate_scripture(
        state,
        resolve_selection_for_single_book=_resolve_selection_for_single_book,
        resolve_bible_data_root=resolve_bible_data_root,
        list_available_sources=list_available_sources,
        select_verses=select_verses,
        label_ranges=label_ranges,
        translate_text=translate_text,
        open_ai_client=open_ai_client,
        add_tokens=add_tokens,
        extract_cached_input_tokens=_extract_cached_input_tokens,
        TranslatedPassage=TranslatedPassage,
        ResponseLanguage=ResponseLanguage,
        TARGET_TRANSLATION_LANGUAGE_AGENT_SYSTEM_PROMPT=(
            TARGET_TRANSLATION_LANGUAGE_AGENT_SYSTEM_PROMPT
        ),
        TRANSLATE_PASSAGE_AGENT_SYSTEM_PROMPT=(
            TRANSLATE_PASSAGE_AGENT_SYSTEM_PROMPT
        ),
        supported_language_map=supported_language_map,
    )


def create_brain():
    """Compile the LangGraph using the split graph helper."""

    nodes = {
        "start_node": start,
        "determine_query_language_node": determine_query_language,
        "preprocess_user_query_node": preprocess_user_query,
        "determine_intents_node": determine_intents,
        "set_response_language_node": set_response_language,
        "query_vector_db_node": query_vector_db,
        "query_open_ai_node": query_open_ai,
        "chunk_message_node": chunk_message,
        "handle_unsupported_function_node": handle_unsupported_function,
        "handle_system_information_request_node": handle_system_information_request,
        "converse_with_bt_servant_node": converse_with_bt_servant,
        "get_passage_summary_node": get_passage_summary,
        "get_passage_keywords_node": get_passage_keywords,
        "get_translation_helps_node": get_translation_helps,
        "retrieve_scripture_node": retrieve_scripture,
        "listen_to_scripture_node": listen_to_scripture,
        "translate_scripture_node": translate_scripture,
        "translate_responses_node": translate_responses,
        "needs_chunking": needs_chunking,
    }
    return _graph_create(BrainStateType=BrainState, nodes=nodes)


__all__ = [
    "BrainState",
    "Capability",
    "IntentType",
    "UserIntents",
    "Language",
    "ResponseLanguage",
    "MessageLanguage",
    "TranslatedPassage",
    "PassageSelection",
    "PassageRef",
    "config",
    "open_ai_client",
    "supported_language_map",
    "LANGUAGE_UNKNOWN",
    "RELEVANCE_CUTOFF",
    "TOP_K",
    "capabilities",
    "build_boilerplate_message",
    "build_full_help_message",
    "BOILER_PLATE_AVAILABLE_FEATURES_MESSAGE",
    "FULL_HELP_MESSAGE",
    "FIRST_INTERACTION_MESSAGE",
    "CONVERSE_AGENT_SYSTEM_PROMPT",
    "HELP_AGENT_SYSTEM_PROMPT",
    "UNSUPPORTED_FUNCTION_AGENT_SYSTEM_PROMPT",
    "CHOP_AGENT_SYSTEM_PROMPT",
    "COMBINE_RESPONSES_SYSTEM_PROMPT",
    "RESPONSE_TRANSLATOR_SYSTEM_PROMPT",
    "TRANSLATE_PASSAGE_AGENT_SYSTEM_PROMPT",
    "DETECT_LANGUAGE_AGENT_SYSTEM_PROMPT",
    "SET_RESPONSE_LANGUAGE_AGENT_SYSTEM_PROMPT",
    "TARGET_TRANSLATION_LANGUAGE_AGENT_SYSTEM_PROMPT",
    "PREPROCESSOR_AGENT_SYSTEM_PROMPT",
    "PreprocessorResult",
    "start",
    "detect_language",
    "determine_query_language",
    "preprocess_user_query",
    "determine_intents",
    "set_response_language",
    "query_vector_db",
    "query_open_ai",
    "chunk_message",
    "needs_chunking",
    "handle_unsupported_function",
    "handle_system_information_request",
    "converse_with_bt_servant",
    "get_passage_summary",
    "get_passage_keywords",
    "get_translation_helps",
    "retrieve_scripture",
    "listen_to_scripture",
    "translate_scripture",
    "translate_responses",
    "process_intents",
    "combine_responses",
    "translate_text",
    "create_brain",
]
