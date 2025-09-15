"""Core state and capability types for the brain graph.

Extracted from the legacy monolithic brain.py to keep orchestration lean.
"""

from __future__ import annotations

import operator
from typing import Any, Dict, List, Annotated
from typing_extensions import TypedDict


class BrainState(TypedDict, total=False):
    """State carried through the LangGraph execution."""

    user_id: str
    user_query: str
    # Perf tracing: preserve trace id throughout the graph so node wrappers
    # can attach spans even when running in a thread pool.
    perf_trace_id: str
    query_language: str
    user_response_language: str
    transformed_query: str
    docs: List[Dict[str, str]]
    collection_used: str
    responses: Annotated[List[Dict[str, Any]], operator.add]
    translated_responses: List[str]
    stack_rank_collections: List[str]
    user_chat_history: List[Dict[str, str]]
    # Downstream intent classifier output
    user_intents: List[Any]
    passage_selection: list[dict]
    # Delivery hint for bt_servant to send a voice message instead of text
    send_voice_message: bool


class Capability(TypedDict):
    """User-facing capability metadata for feature listings and help text."""

    intent: str
    label: str
    description: str
    examples: List[str]
    include_in_boilerplate: bool
