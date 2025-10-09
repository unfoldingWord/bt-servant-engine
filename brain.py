"""BT Servant brain - decision graph and message processing pipeline.

This module serves as the public API for the brain subsystem. The actual
implementations have been extracted to service modules:
- brain_nodes: All graph node implementations
- brain_orchestrator: LangGraph setup and routing
- graph_pipeline: Vector DB and RAG queries
- response_pipeline: Translation and chunking
- preprocessing: Query processing and intent classification
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Dict, List

from typing_extensions import TypedDict

from bt_servant_engine.core.config import config
from bt_servant_engine.core.intents import IntentType

# Re-export node functions for backward compatibility
from bt_servant_engine.services.brain_nodes import (
    open_ai_client,
    get_chroma_collection,
    set_user_agentic_strength,
    set_user_response_language,
    FIA_REFERENCE_CONTENT,
    resolve_selection_for_single_book,
    chunk_message,
    combine_responses,
    consult_fia_resources,
    converse_with_bt_servant,
    determine_intents,
    determine_query_language,
    handle_get_passage_keywords,
    handle_get_passage_summary,
    handle_get_translation_helps,
    handle_listen_to_scripture,
    handle_retrieve_scripture,
    handle_system_information_request,
    handle_translate_scripture,
    handle_unsupported_function,
    needs_chunking,
    preprocess_user_query,
    query_open_ai,
    query_vector_db,
    set_agentic_strength,
    set_response_language,
    start,
    translate_responses,
    translate_text,
)

# Re-export orchestration
from bt_servant_engine.services.brain_orchestrator import create_brain


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


__all__ = [
    # State
    "BrainState",
    # Orchestration
    "create_brain",
    # Dependencies (for test compatibility)
    "config",
    "IntentType",
    "open_ai_client",
    "get_chroma_collection",
    "set_user_agentic_strength",
    "set_user_response_language",
    "FIA_REFERENCE_CONTENT",
    "resolve_selection_for_single_book",
    # Node functions
    "start",
    "determine_intents",
    "set_response_language",
    "set_agentic_strength",
    "combine_responses",
    "translate_responses",
    "translate_text",
    "determine_query_language",
    "preprocess_user_query",
    "query_vector_db",
    "query_open_ai",
    "consult_fia_resources",
    "chunk_message",
    "needs_chunking",
    "handle_unsupported_function",
    "handle_system_information_request",
    "converse_with_bt_servant",
    "handle_get_passage_summary",
    "handle_get_passage_keywords",
    "handle_get_translation_helps",
    "handle_retrieve_scripture",
    "handle_listen_to_scripture",
    "handle_translate_scripture",
]
