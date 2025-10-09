"""Decision graph and message-processing pipeline for BT Servant.

This module defines the state, nodes, and orchestration logic for handling
incoming user messages, classifying intents, querying resources, and producing
final responses (including translation and chunking when necessary).
"""
# pylint: disable=line-too-long,too-many-lines,too-many-statements

from __future__ import annotations

import operator
from pathlib import Path
from typing import Annotated, Any, Dict, Iterable, List, Optional, cast

from openai import OpenAI
from typing_extensions import TypedDict

from bt_servant_engine.core.config import config
from bt_servant_engine.core.intents import IntentType
from bt_servant_engine.core.language import (
    SUPPORTED_LANGUAGE_MAP as supported_language_map,
)
from bt_servant_engine.core.logging import get_logger
from bt_servant_engine.services.openai_utils import (
    extract_cached_input_tokens as _extract_cached_input_tokens,
)
from bt_servant_engine.services.passage_helpers import (
    book_patterns as _book_patterns_impl,
    choose_primary_book as _choose_primary_book_impl,
    detect_mentioned_books as _detect_mentioned_books_impl,
)
from bt_servant_engine.services.response_helpers import (
    is_protected_response_item as _is_protected_response_item_impl,
    normalize_single_response as _normalize_single_response_impl,
    partition_response_items as _partition_response_items_impl,
    sample_for_language_detection as _sample_for_language_detection_impl,
)
from bt_servant_engine.services.preprocessing import (
    determine_intents as _determine_intents_impl,
    determine_query_language as _determine_query_language_impl,
    model_for_agentic_strength as _model_for_agentic_strength,
    preprocess_user_query as _preprocess_user_query_impl,
    resolve_agentic_strength as _resolve_agentic_strength,
)
from bt_servant_engine.services.passage_selection import (
    resolve_selection_for_single_book as _resolve_selection_for_single_book_impl,
)
from bt_servant_engine.services.intents.simple_intents import (
    BOILER_PLATE_AVAILABLE_FEATURES_MESSAGE,
    converse_with_bt_servant as converse_with_bt_servant_impl,
    handle_system_information_request as handle_system_information_request_impl,
    handle_unsupported_function as handle_unsupported_function_impl,
)
from bt_servant_engine.services.intents.settings_intents import (
    set_agentic_strength as set_agentic_strength_impl,
    set_response_language as set_response_language_impl,
)
from bt_servant_engine.services.intents.passage_intents import (
    get_passage_keywords as get_passage_keywords_impl,
    get_passage_summary as get_passage_summary_impl,
    listen_to_scripture as listen_to_scripture_impl,
    retrieve_scripture as retrieve_scripture_impl,
)
from bt_servant_engine.services.intents.translation_intents import (
    get_translation_helps as get_translation_helps_impl,
    translate_scripture as translate_scripture_impl,
)
from bt_servant_engine.services.intents.fia_intents import (
    consult_fia_resources as consult_fia_resources_impl,
)
from bt_servant_engine.services.response_pipeline import (
    chunk_message as chunk_message_impl,
    combine_responses as combine_responses_impl,
    needs_chunking as needs_chunking_impl,
    reconstruct_structured_text as reconstruct_structured_text_impl,
    translate_or_localize_response as translate_or_localize_response_impl,
    translate_text as translate_text_impl,
    build_translation_queue as build_translation_queue_impl,
    resolve_target_language as resolve_target_language_impl,
)
from bt_servant_engine.services.graph_pipeline import (
    query_open_ai as query_open_ai_impl,
    query_vector_db as query_vector_db_impl,
)
from bt_servant_engine.services.translation_helpers import (
    build_translation_helps_context as build_translation_helps_context_impl,
    build_translation_helps_messages as build_translation_helps_messages_impl,
    prepare_translation_helps as prepare_translation_helps_impl,
)
from bt_servant_engine.adapters.chroma import get_chroma_collection
from bt_servant_engine.adapters.user_state import (
    is_first_interaction,
    set_first_interaction,
    set_user_agentic_strength,
    set_user_response_language,
)
from utils.bsb import (
    BOOK_MAP as BSB_BOOK_MAP,
)
from utils.perf import add_tokens

# Import LangGraph orchestration from brain_orchestrator
# Note: This creates a circular dependency at import time, but brain_orchestrator
# imports brain dynamically within create_brain() to break the cycle
from bt_servant_engine.services.brain_orchestrator import (
    create_brain,
)

# (Moved dynamic feature messaging and related prompts below IntentType)
# COMBINE_RESPONSES_SYSTEM_PROMPT moved to bt_servant_engine.services.response_pipeline


# RESPONSE_TRANSLATOR_SYSTEM_PROMPT moved to bt_servant_engine.services.response_pipeline


# PREPROCESSOR_AGENT_SYSTEM_PROMPT moved to bt_servant_engine.services.preprocessing

# PASSAGE_SELECTION_AGENT_SYSTEM_PROMPT moved to bt_servant_engine.services.passage_selection


# FINAL_RESPONSE_AGENT_SYSTEM_PROMPT moved to bt_servant_engine.services.graph_pipeline

# CONSULT_FIA_RESOURCES_SYSTEM_PROMPT moved to bt_servant_engine.services.intents.fia_intents

# CHOP_AGENT_SYSTEM_PROMPT moved to bt_servant_engine.services.response_pipeline

# INTENT_CLASSIFICATION_AGENT_SYSTEM_PROMPT moved to bt_servant_engine.services.preprocessing

# DETECT_LANGUAGE_AGENT_SYSTEM_PROMPT moved to bt_servant_engine.services.preprocessing


BASE_DIR = Path(__file__).resolve().parent
DB_DIR = config.DATA_DIR
FIA_REFERENCE_PATH = BASE_DIR / "sources" / "fia" / "fia.md"

open_ai_client = OpenAI(api_key=config.OPENAI_API_KEY)

# Language constants imported from bt_servant_engine.core.language
# (supported_language_map, LANGUAGE_UNKNOWN)

# RELEVANCE_CUTOFF and TOP_K moved to bt_servant_engine.services.graph_pipeline

logger = get_logger(__name__)

try:
    FIA_REFERENCE_CONTENT = FIA_REFERENCE_PATH.read_text(encoding="utf-8")
except FileNotFoundError:
    logger.warning("FIA reference file missing at %s", FIA_REFERENCE_PATH)
    FIA_REFERENCE_CONTENT = ""


# _extract_cached_input_tokens imported from bt_servant_engine.services.openai_utils


# Language, ResponseLanguage imported from bt_servant_engine.core.language


# SET_RESPONSE_LANGUAGE_AGENT_SYSTEM_PROMPT moved to bt_servant_engine.services.intents.settings_intents
# SET_AGENTIC_STRENGTH_AGENT_SYSTEM_PROMPT moved to bt_servant_engine.services.intents.settings_intents


# _resolve_agentic_strength moved to bt_servant_engine.services.preprocessing (as resolve_agentic_strength)

# _model_for_agentic_strength moved to bt_servant_engine.services.preprocessing (as model_for_agentic_strength)

# MessageLanguage, TranslatedPassage imported from bt_servant_engine.core.language

# PreprocessorResult moved to bt_servant_engine.services.preprocessing


def _is_protected_response_item(item: dict) -> bool:
    """Return True if a response item carries scripture to protect from changes."""
    return _is_protected_response_item_impl(item)


def _reconstruct_structured_text(resp_item: dict | str, localize_to: Optional[str]) -> str:
    """Render a response item to plain text, optionally localizing the header book name."""
    return reconstruct_structured_text_impl(resp_item, localize_to)


def _partition_response_items(responses: Iterable[dict]) -> tuple[list[dict], list[dict]]:
    """Split responses into scripture-protected and normal sets."""
    return _partition_response_items_impl(responses)


def _normalize_single_response(item: dict) -> dict | str:
    """Return a representation suitable for translation when no combine is needed."""
    return _normalize_single_response_impl(item)


def _build_translation_queue(
    state: BrainState,
    protected_items: list[dict],
    normal_items: list[dict],
) -> list[dict | str]:
    """Assemble responses in the order they should be translated or localized."""
    return build_translation_queue_impl(state, protected_items, normal_items, combine_responses)


def _resolve_target_language(
    state: BrainState,
    responses_for_translation: list[dict | str],
) -> tuple[Optional[str], Optional[list[str]]]:
    """Determine the target language or build a pass-through fallback."""
    return resolve_target_language_impl(state, responses_for_translation)


def _translate_or_localize_response(
    resp: dict | str,
    target_language: str,
    agentic_strength: str,
) -> str:
    """Translate free-form text or localize structured scripture outputs."""
    return translate_or_localize_response_impl(
        open_ai_client,
        resp,
        target_language,
        agentic_strength,
        _model_for_agentic_strength,
        _extract_cached_input_tokens,
    )


# IntentType, UserIntents imported from bt_servant_engine.core.intents


class BrainState(TypedDict, total=False):
    """State carried through the LangGraph execution."""

    user_id: str
    user_query: str
    # Perf tracing: preserve trace id throughout the graph so node wrappers
    # can attach spans even when running in a thread pool.
    perf_trace_id: str
    query_language: str
    user_response_language: str
    agentic_strength: str
    user_agentic_strength: str
    transformed_query: str
    docs: List[Dict[str, str]]
    collection_used: str
    responses: Annotated[List[Dict[str, Any]], operator.add]
    translated_responses: List[str]
    stack_rank_collections: List[str]
    user_chat_history: List[Dict[str, str]]
    user_intents: List[IntentType]
    passage_selection: list[dict]
    # Delivery hint for bt_servant to send a voice message instead of text
    send_voice_message: bool
    voice_message_text: str


# Centralized capability registry and builders for feature help/boilerplate
# Capability class moved to bt_servant_engine.services.intents.simple_intents
# Imported above as part of simple_intents imports

# _capabilities moved to bt_servant_engine.services.intents.simple_intents (as get_capabilities)
# Imported above as _capabilities

# BOILER_PLATE_AVAILABLE_FEATURES_MESSAGE and FULL_HELP_MESSAGE imported from simple_intents above

FIRST_INTERACTION_MESSAGE = f"""
Hello! I am the BT Servant. This is our first conversation. Let's work together to understand and translate God's word!

{BOILER_PLATE_AVAILABLE_FEATURES_MESSAGE}
"""


def start(state: Any) -> dict:
    """Handle first interaction greeting, otherwise no-op."""
    s = cast(BrainState, state)
    user_id = s["user_id"]
    if is_first_interaction(user_id):
        set_first_interaction(user_id, False)
        return {
            "responses": [{"intent": "first-interaction", "response": FIRST_INTERACTION_MESSAGE}]
        }
    return {}


def determine_intents(state: Any) -> dict:
    """Classify the user's transformed query into one or more intents."""
    s = cast(BrainState, state)
    query = s["transformed_query"]
    user_intents = _determine_intents_impl(open_ai_client, query)
    return {
        "user_intents": user_intents,
    }


def set_response_language(state: Any) -> dict:
    """Detect and persist the user's desired response language."""
    s = cast(BrainState, state)
    return set_response_language_impl(
        open_ai_client,
        s["user_id"],
        s["user_query"],
        s["user_chat_history"],
        supported_language_map,
        set_user_response_language,
    )


def set_agentic_strength(state: Any) -> dict:
    """Detect and persist the user's preferred agentic strength."""
    s = cast(BrainState, state)
    return set_agentic_strength_impl(
        open_ai_client,
        s["user_id"],
        s["user_query"],
        s["user_chat_history"],
        set_user_agentic_strength,
        config.LOG_PSEUDONYM_SECRET,
    )


def combine_responses(chat_history, latest_user_message, responses) -> str:
    """Ask OpenAI to synthesize multiple node responses into one coherent text."""
    return combine_responses_impl(
        open_ai_client,
        chat_history,
        latest_user_message,
        responses,
        _extract_cached_input_tokens,
    )


def translate_responses(state: Any) -> dict:
    """Translate or localize responses into the user's desired language."""
    s = cast(BrainState, state)
    raw_responses = [
        resp for resp in cast(list[dict], s["responses"]) if not resp.get("suppress_text_delivery")
    ]
    if not raw_responses:
        if bool(s.get("send_voice_message")):
            logger.info("[translate] skipping text translation because delivery is voice-only")
            return {"translated_responses": []}
        raise ValueError("no responses to translate. something bad happened. bailing out.")

    protected_items, normal_items = _partition_response_items(raw_responses)
    responses_for_translation = _build_translation_queue(s, protected_items, normal_items)
    if not responses_for_translation:
        if bool(s.get("send_voice_message")):
            logger.info("[translate] no text responses after queue assembly; voice-only delivery")
            return {"translated_responses": []}
        raise ValueError("no responses to translate. something bad happened. bailing out.")

    target_language, passthrough = _resolve_target_language(s, responses_for_translation)
    if passthrough is not None:
        return {"translated_responses": passthrough}
    assert target_language is not None

    agentic_strength = _resolve_agentic_strength(s)
    translated_responses = [
        _translate_or_localize_response(resp, target_language, agentic_strength)
        for resp in responses_for_translation
    ]
    return {"translated_responses": translated_responses}


def translate_text(
    response_text: str,
    target_language: str,
    *,
    agentic_strength: Optional[str] = None,
) -> str:
    """Translate a single text into the target ISO 639-1 language code."""
    return translate_text_impl(
        open_ai_client,
        response_text,
        target_language,
        _model_for_agentic_strength,
        _extract_cached_input_tokens,
        agentic_strength=agentic_strength,
    )


# detect_language moved to bt_servant_engine.services.preprocessing


def determine_query_language(state: Any) -> dict:
    """Determine the language of the user's original query and set collection order."""
    s = cast(BrainState, state)
    query = s["user_query"]
    agentic_strength = _resolve_agentic_strength(s)
    query_language, stack_rank_collections = _determine_query_language_impl(
        open_ai_client, query, agentic_strength
    )
    return {"query_language": query_language, "stack_rank_collections": stack_rank_collections}


def preprocess_user_query(state: Any) -> dict:
    """Lightly clarify or correct the user's query using conversation history."""
    s = cast(BrainState, state)
    query = s["user_query"]
    chat_history = s["user_chat_history"]
    transformed_query, _, _ = _preprocess_user_query_impl(open_ai_client, query, chat_history)
    return {"transformed_query": transformed_query}


def query_vector_db(state: Any) -> dict:
    """Query the vector DB (Chroma) across ranked collections and filter by relevance."""
    s = cast(BrainState, state)
    return query_vector_db_impl(
        s["transformed_query"],
        s["stack_rank_collections"],
        get_chroma_collection,
        BOILER_PLATE_AVAILABLE_FEATURES_MESSAGE,
    )


def query_open_ai(state: Any) -> dict:
    """Generate the final response text using RAG context and OpenAI."""
    s = cast(BrainState, state)
    agentic_strength = _resolve_agentic_strength(s)
    return query_open_ai_impl(
        open_ai_client,
        s["docs"],
        s["transformed_query"],
        s["user_chat_history"],
        _model_for_agentic_strength,
        _extract_cached_input_tokens,
        add_tokens,
        agentic_strength,
        BOILER_PLATE_AVAILABLE_FEATURES_MESSAGE,
    )


def consult_fia_resources(state: Any) -> dict:
    """Answer FIA-specific questions using FIA collections and reference material."""
    s = cast(BrainState, state)
    agentic_strength = _resolve_agentic_strength(s)
    return consult_fia_resources_impl(
        open_ai_client,
        s["transformed_query"],
        s["user_chat_history"],
        s.get("user_response_language"),
        s.get("query_language"),
        get_chroma_collection,
        _model_for_agentic_strength,
        _extract_cached_input_tokens,
        agentic_strength,
    )


def chunk_message(state: Any) -> dict:
    """Chunk oversized responses to respect WhatsApp limits, via LLM or fallback."""
    s = cast(BrainState, state)
    responses = s["translated_responses"]
    text_to_chunk = responses[0]
    chunk_max = config.MAX_META_TEXT_LENGTH - 100
    chunks = chunk_message_impl(
        open_ai_client,
        text_to_chunk,
        _extract_cached_input_tokens,
        responses[1:],
        chunk_max,
    )
    return {"translated_responses": chunks}


def needs_chunking(state: BrainState) -> str:
    """Return next node key if chunking is required, otherwise finish."""
    return needs_chunking_impl(state)


def handle_unsupported_function(state: Any) -> dict:
    """Generate a helpful response when the user requests unsupported functionality."""
    s = cast(BrainState, state)
    return handle_unsupported_function_impl(open_ai_client, s["user_query"], s["user_chat_history"])


def handle_system_information_request(state: Any) -> dict:
    """Provide help/about information for the BT Servant system."""
    s = cast(BrainState, state)
    return handle_system_information_request_impl(
        open_ai_client, s["user_query"], s["user_chat_history"]
    )


def converse_with_bt_servant(state: Any) -> dict:
    """Respond conversationally to the user based on context and history."""
    s = cast(BrainState, state)
    return converse_with_bt_servant_impl(open_ai_client, s["user_query"], s["user_chat_history"])


def _book_patterns() -> list[tuple[str, str]]:
    """Return (canonical, regex) patterns to detect book mentions (ordered)."""
    return _book_patterns_impl(BSB_BOOK_MAP)


def _detect_mentioned_books(text: str) -> list[str]:
    """Detect canonical books mentioned in text, preserving order of appearance."""
    return _detect_mentioned_books_impl(text, BSB_BOOK_MAP)


def _choose_primary_book(text: str, candidates: list[str]) -> str | None:
    """Heuristic to pick a primary book when multiple are mentioned.

    Prefer the first mentioned that appears near chapter/verse digits; else None.
    """
    return _choose_primary_book_impl(text, candidates, BSB_BOOK_MAP)


def _resolve_selection_for_single_book(
    query: str,
    query_lang: str,
    focus_hint: str | None = None,
) -> tuple[str | None, list[tuple[int, int | None, int | None, int | None]] | None, str | None]:
    """Parse and normalize a user query into a single canonical book and ranges."""
    return _resolve_selection_for_single_book_impl(
        client=open_ai_client,
        query=query,
        query_lang=query_lang,
        book_map=BSB_BOOK_MAP,
        detect_mentioned_books_fn=_detect_mentioned_books,
        translate_text_fn=translate_text,
        focus_hint=focus_hint,
    )


def handle_get_passage_summary(state: Any) -> dict:
    """Handle get-passage-summary: extract refs, retrieve verses, summarize."""
    s = cast(BrainState, state)
    agentic_strength = _resolve_agentic_strength(s)
    return get_passage_summary_impl(
        open_ai_client,
        s["transformed_query"],
        s["query_language"],
        BSB_BOOK_MAP,
        _detect_mentioned_books,
        translate_text,
        _model_for_agentic_strength,
        _extract_cached_input_tokens,
        s.get("user_response_language"),
        agentic_strength,
    )


def handle_get_passage_keywords(state: Any) -> dict:
    """Handle get-passage-keywords: extract refs, retrieve keywords, and list them."""
    s = cast(BrainState, state)
    return get_passage_keywords_impl(
        open_ai_client,
        s["transformed_query"],
        s["query_language"],
        BSB_BOOK_MAP,
        _detect_mentioned_books,
        translate_text,
    )


def handle_get_translation_helps(state: Any) -> dict:
    """Generate focused translation helps guidance for a selected passage."""
    s = cast(BrainState, state)
    agentic_strength = _resolve_agentic_strength(s)
    return get_translation_helps_impl(
        open_ai_client,
        s["transformed_query"],
        s["query_language"],
        BSB_BOOK_MAP,
        _detect_mentioned_books,
        translate_text,
        _model_for_agentic_strength,
        _extract_cached_input_tokens,
        prepare_translation_helps_impl,
        build_translation_helps_context_impl,
        build_translation_helps_messages_impl,
        agentic_strength,
    )


def handle_retrieve_scripture(state: Any) -> dict:
    """Handle retrieve-scripture with optional auto-translation."""
    s = cast(BrainState, state)
    agentic_strength = _resolve_agentic_strength(s)
    return retrieve_scripture_impl(
        open_ai_client,
        s["transformed_query"],
        s["query_language"],
        BSB_BOOK_MAP,
        _detect_mentioned_books,
        translate_text,
        _model_for_agentic_strength,
        _extract_cached_input_tokens,
        s.get("user_response_language"),
        agentic_strength,
    )


def handle_listen_to_scripture(state: Any) -> dict:
    """Delegate to retrieve-scripture and request voice delivery."""
    s = cast(BrainState, state)
    agentic_strength = _resolve_agentic_strength(s)
    return listen_to_scripture_impl(
        open_ai_client,
        s["transformed_query"],
        s["query_language"],
        BSB_BOOK_MAP,
        _detect_mentioned_books,
        translate_text,
        _reconstruct_structured_text,
        _model_for_agentic_strength,
        _extract_cached_input_tokens,
        s.get("user_response_language"),
        agentic_strength,
    )


def handle_translate_scripture(state: Any) -> dict:
    """Handle translate-scripture: return verses translated into a target language."""
    s = cast(BrainState, state)
    agentic_strength = _resolve_agentic_strength(s)
    return translate_scripture_impl(
        open_ai_client,
        s["transformed_query"],
        s["query_language"],
        BSB_BOOK_MAP,
        _detect_mentioned_books,
        translate_text,
        _model_for_agentic_strength,
        _extract_cached_input_tokens,
        s.get("user_response_language"),
        agentic_strength,
    )


LANG_DETECTION_SAMPLE_CHARS = 100


def _sample_for_language_detection(text: str) -> str:
    """Return a short prefix ending at a whitespace boundary for detection."""
    return _sample_for_language_detection_impl(text, LANG_DETECTION_SAMPLE_CHARS)
