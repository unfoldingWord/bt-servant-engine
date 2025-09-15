"""Graph wiring helpers for the BT Servant brain.

Exposes small, dependency-light helpers for process_intents and graph
assembly; the orchestration module passes node functions in to avoid
cross-imports and cycles.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Type, cast
from collections.abc import Hashable

from langgraph.graph import StateGraph

from utils.perf import time_block, set_current_trace
from servant_brain.classifier import IntentType


def process_intents(state: Any) -> List[Hashable]:  # pylint: disable=too-many-branches  # noqa: C901
    """Map detected intents to the list of nodes to traverse."""
    s = cast(Dict[str, Any], state)
    intents: List[IntentType] = cast(List[IntentType], s.get("user_intents", []))
    nodes_to_traverse: List[Hashable] = []
    if not intents:
        return ["handle_unsupported_function_node"]

    # Route single-intent or multi-intent flows.
    for intent in intents:
        if intent == IntentType.SET_RESPONSE_LANGUAGE:
            nodes_to_traverse.append("set_response_language_node")
        elif intent == IntentType.PERFORM_UNSUPPORTED_FUNCTION:
            nodes_to_traverse.append("handle_unsupported_function_node")
        elif intent == IntentType.RETRIEVE_SYSTEM_INFORMATION:
            nodes_to_traverse.append("handle_system_information_request_node")
        elif intent == IntentType.CONVERSE_WITH_BT_SERVANT:
            nodes_to_traverse.append("converse_with_bt_servant_node")
        elif intent == IntentType.GET_PASSAGE_SUMMARY:
            nodes_to_traverse.append("get_passage_summary_node")
        elif intent == IntentType.GET_PASSAGE_KEYWORDS:
            nodes_to_traverse.append("get_passage_keywords_node")
        elif intent == IntentType.GET_TRANSLATION_HELPS:
            nodes_to_traverse.append("get_translation_helps_node")
        elif intent == IntentType.RETRIEVE_SCRIPTURE:
            nodes_to_traverse.append("retrieve_scripture_node")
        elif intent == IntentType.LISTEN_TO_SCRIPTURE:
            nodes_to_traverse.append("listen_to_scripture_node")
        elif intent == IntentType.TRANSLATE_SCRIPTURE:
            nodes_to_traverse.append("translate_scripture_node")
        else:
            nodes_to_traverse.append("query_vector_db_node")

    if not nodes_to_traverse:
        nodes_to_traverse = ["query_vector_db_node"]
    return nodes_to_traverse


def create_brain(
    *,
    BrainStateType: Type[Any],
    nodes: Dict[str, Callable[[Any], dict]],
) -> Any:
    """Assemble and compile the LangGraph for the brain.

    The caller provides the BrainState type and a mapping of node names to
    functions. This keeps this module free of orchestrator details and avoids
    import cycles.
    """

    def wrap_node_with_timing(node_fn: Callable[[Any], dict], node_name: str):
        def wrapped(state: Any) -> dict:
            trace_id = cast(Dict[str, Any], state).get("perf_trace_id")
            if trace_id:
                set_current_trace(cast(Optional[str], trace_id))
            with time_block(f"brain:{node_name}"):
                return node_fn(state)
        return wrapped

    builder: StateGraph[Any] = StateGraph(BrainStateType)

    # Register nodes
    for name, fn in nodes.items():
        if name == "translate_responses_node":
            builder.add_node(name, wrap_node_with_timing(fn, name), defer=True)
        else:
            builder.add_node(name, wrap_node_with_timing(fn, name))

    # Edges and flow
    builder.set_entry_point("start_node")
    builder.add_edge("start_node", "determine_query_language_node")
    builder.add_edge("determine_query_language_node", "preprocess_user_query_node")
    builder.add_edge("preprocess_user_query_node", "determine_intents_node")
    builder.add_conditional_edges("determine_intents_node", process_intents)
    builder.add_edge("query_vector_db_node", "query_open_ai_node")
    builder.add_edge("set_response_language_node", "translate_responses_node")

    builder.add_edge("handle_unsupported_function_node", "translate_responses_node")
    builder.add_edge("handle_system_information_request_node", "translate_responses_node")
    builder.add_edge("converse_with_bt_servant_node", "translate_responses_node")
    builder.add_edge("get_passage_summary_node", "translate_responses_node")
    builder.add_edge("get_passage_keywords_node", "translate_responses_node")
    builder.add_edge("get_translation_helps_node", "translate_responses_node")
    builder.add_edge("retrieve_scripture_node", "translate_responses_node")
    builder.add_edge("listen_to_scripture_node", "translate_responses_node")
    builder.add_edge("translate_scripture_node", "translate_responses_node")
    builder.add_edge("query_open_ai_node", "translate_responses_node")

    # The conditional function returns a node label string or list of labels.
    # Split across lines to satisfy line length.
    builder.add_conditional_edges(
        "translate_responses_node",
        nodes["needs_chunking"],  # type: ignore[arg-type]
    )
    builder.set_finish_point("chunk_message_node")

    return builder.compile()
