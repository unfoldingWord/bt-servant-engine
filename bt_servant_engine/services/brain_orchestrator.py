"""LangGraph orchestration setup for BT Servant brain.

This module contains the graph construction, node routing, and timing infrastructure
for the brain decision pipeline.
"""

from __future__ import annotations

import operator
from collections.abc import Hashable
from typing import TYPE_CHECKING, Annotated, Any, Awaitable, Callable, Dict, List, Optional, cast

from langgraph.graph import StateGraph
from typing_extensions import TypedDict

from bt_servant_engine.core.intents import IntentType
from utils.perf import set_current_trace, time_block

if TYPE_CHECKING:
    pass


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
    docs: List[Dict[str, Any]]
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
    # Progress messaging fields
    progress_enabled: bool
    progress_messenger: Optional[Callable[[str], Awaitable[None]]]
    last_progress_time: float
    progress_throttle_seconds: float


ProgressMessageInput = str | Callable[[Any], Optional[str]]
BASE_TRANSLATION_ASSISTANCE_PROGRESS_MESSAGE = (
    "I'm pulling everything together into a helpful response for you."
)


def wrap_node_with_timing(node_fn, node_name: str):  # type: ignore[no-untyped-def]
    """Wrap a node function with performance timing and tracing."""

    def wrapped(state: Any) -> dict:
        trace_id = cast(dict, state).get("perf_trace_id")
        if trace_id:
            set_current_trace(cast(Optional[str], trace_id))
        with time_block(f"brain:{node_name}"):
            return node_fn(state)

    return wrapped


def wrap_node_with_progress(  # type: ignore[no-untyped-def]
    node_fn,
    node_name: str,
    progress_message: ProgressMessageInput | None = None,
    condition: Optional[Callable[[Any], bool]] = None,
    force: bool = False,
):
    """Wrap a node with timing and optional progress messaging.

    Args:
        node_fn: The node function to wrap
        node_name: Name of the node for timing instrumentation
        progress_message: Optional message to send before node execution
        condition: Optional callable to determine if progress should be shown
        force: If True, bypass throttling for this message

    Returns:
        Wrapped node function with progress messaging support
    """
    import asyncio

    def wrapped(state: Any) -> dict:
        # Send progress message before node execution if configured
        should_show = condition is None or condition(state)
        message_text: Optional[str] = None
        if progress_message is not None and should_show:
            message_text = (
                progress_message(state) if callable(progress_message) else progress_message
            )

        if message_text:
            # Import here to avoid circular dependency
            from bt_servant_engine.services.progress_messaging import maybe_send_progress

            coroutine = maybe_send_progress(state, message_text, force=force)
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                # No running loop in this thread (common when inside a thread pool)
                asyncio.run(coroutine)
            else:
                # Already inside an event loop; schedule without blocking to avoid deadlocks.
                loop.create_task(coroutine)

        # Execute the original node with timing
        trace_id = cast(dict, state).get("perf_trace_id")
        if trace_id:
            set_current_trace(cast(Optional[str], trace_id))
        with time_block(f"brain:{node_name}"):
            return node_fn(state)

    return wrapped


def _format_series(resources: List[str]) -> str:
    if not resources:
        return ""
    if len(resources) == 1:
        return resources[0]
    if len(resources) == 2:
        return f"{resources[0]} and {resources[1]}"
    return f"{', '.join(resources[:-1])}, and {resources[-1]}"


def _collect_resource_sources(docs: List[Dict[str, Any]]) -> List[str]:
    seen: set[str] = set()
    ordered: List[str] = []
    for doc in docs:
        metadata = doc.get("metadata")
        if not isinstance(metadata, dict):
            continue
        merged_from = metadata.get("_merged_from")
        if not isinstance(merged_from, str):
            continue
        normalized = merged_from.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return ordered


def build_translation_assistance_progress_message(state: Any) -> Optional[str]:
    """Generate the progress message for translation assistance summarizing resource origins."""
    s = cast(BrainState, state)
    docs = cast(List[Dict[str, Any]], s.get("docs", []))
    sources = _collect_resource_sources(docs)
    if not sources:
        return BASE_TRANSLATION_ASSISTANCE_PROGRESS_MESSAGE

    resources_list = _format_series(sources)
    return (
        "I found potentially relevant documents in the following resources: "
        f"{resources_list}. {BASE_TRANSLATION_ASSISTANCE_PROGRESS_MESSAGE}"
    )


def process_intents(state: Any) -> List[Hashable]:  # pylint: disable=too-many-branches
    """Map detected intents to the list of nodes to traverse."""
    s = cast(BrainState, state)
    user_intents = s["user_intents"]
    if not user_intents:
        raise ValueError("no intents found. something went very wrong.")

    nodes_to_traverse: List[Hashable] = []
    if IntentType.GET_BIBLE_TRANSLATION_ASSISTANCE in user_intents:
        nodes_to_traverse.append("query_vector_db_node")
    if IntentType.CONSULT_FIA_RESOURCES in user_intents:
        nodes_to_traverse.append("consult_fia_resources_node")
    if IntentType.GET_PASSAGE_SUMMARY in user_intents:
        nodes_to_traverse.append("handle_get_passage_summary_node")
    if IntentType.GET_PASSAGE_KEYWORDS in user_intents:
        nodes_to_traverse.append("handle_get_passage_keywords_node")
    if IntentType.GET_TRANSLATION_HELPS in user_intents:
        nodes_to_traverse.append("handle_get_translation_helps_node")
    if IntentType.RETRIEVE_SCRIPTURE in user_intents:
        nodes_to_traverse.append("handle_retrieve_scripture_node")
    if IntentType.LISTEN_TO_SCRIPTURE in user_intents:
        nodes_to_traverse.append("handle_listen_to_scripture_node")
    if IntentType.SET_RESPONSE_LANGUAGE in user_intents:
        nodes_to_traverse.append("set_response_language_node")
    if IntentType.SET_AGENTIC_STRENGTH in user_intents:
        nodes_to_traverse.append("set_agentic_strength_node")
    if IntentType.PERFORM_UNSUPPORTED_FUNCTION in user_intents:
        nodes_to_traverse.append("handle_unsupported_function_node")
    if IntentType.RETRIEVE_SYSTEM_INFORMATION in user_intents:
        nodes_to_traverse.append("handle_system_information_request_node")
    if IntentType.CONVERSE_WITH_BT_SERVANT in user_intents:
        nodes_to_traverse.append("converse_with_bt_servant_node")
    if IntentType.RETRIEVE_SCRIPTURE in user_intents:
        nodes_to_traverse.append("handle_retrieve_scripture_node")
    if IntentType.LISTEN_TO_SCRIPTURE in user_intents:
        nodes_to_traverse.append("handle_listen_to_scripture_node")
    if IntentType.TRANSLATE_SCRIPTURE in user_intents:
        nodes_to_traverse.append("handle_translate_scripture_node")

    return nodes_to_traverse


def create_brain():
    """Assemble and compile the LangGraph for the BT Servant brain."""
    # Import here to avoid circular dependency
    from bt_servant_engine.services import brain_nodes
    from bt_servant_engine.services.progress_messaging import should_show_translation_progress

    def _make_state_graph(schema: Any) -> StateGraph:
        # Accept Any to satisfy IDE variance on schema param; schema is BrainState
        return StateGraph(schema)

    def _should_show_translation_progress(state: Any) -> bool:
        """Local wrapper for translation progress condition."""
        return should_show_translation_progress(state)

    builder: StateGraph = _make_state_graph(BrainState)

    # Add all nodes with timing wrappers
    builder.add_node(
        "start_node",
        wrap_node_with_progress(
            brain_nodes.start,
            "start_node",
            progress_message="Give me a few moments to think about your message.",
            force=True,
        ),
    )
    builder.add_node(
        "determine_query_language_node",
        wrap_node_with_timing(
            brain_nodes.determine_query_language, "determine_query_language_node"
        ),
    )
    builder.add_node(
        "preprocess_user_query_node",
        wrap_node_with_timing(brain_nodes.preprocess_user_query, "preprocess_user_query_node"),
    )
    builder.add_node(
        "determine_intents_node",
        wrap_node_with_timing(brain_nodes.determine_intents, "determine_intents_node"),
    )
    builder.add_node(
        "set_response_language_node",
        wrap_node_with_timing(brain_nodes.set_response_language, "set_response_language_node"),
    )
    builder.add_node(
        "set_agentic_strength_node",
        wrap_node_with_timing(brain_nodes.set_agentic_strength, "set_agentic_strength_node"),
    )
    builder.add_node(
        "query_vector_db_node",
        wrap_node_with_progress(
            brain_nodes.query_vector_db,
            "query_vector_db_node",
            progress_message="I'm searching various Bible resources to help answer your question.",
            condition=lambda s: IntentType.GET_BIBLE_TRANSLATION_ASSISTANCE
            in s.get("user_intents", []),
            force=True,
        ),
    )
    builder.add_node(
        "query_open_ai_node",
        wrap_node_with_progress(
            brain_nodes.query_open_ai,
            "query_open_ai_node",
            progress_message=build_translation_assistance_progress_message,
            condition=lambda s: IntentType.GET_BIBLE_TRANSLATION_ASSISTANCE
            in s.get("user_intents", []),
            force=True,  # Always send, bypass throttling
        ),
    )
    builder.add_node(
        "consult_fia_resources_node",
        wrap_node_with_timing(brain_nodes.consult_fia_resources, "consult_fia_resources_node"),
    )
    builder.add_node(
        "chunk_message_node", wrap_node_with_timing(brain_nodes.chunk_message, "chunk_message_node")
    )
    builder.add_node(
        "handle_unsupported_function_node",
        wrap_node_with_timing(
            brain_nodes.handle_unsupported_function, "handle_unsupported_function_node"
        ),
    )
    builder.add_node(
        "handle_system_information_request_node",
        wrap_node_with_timing(
            brain_nodes.handle_system_information_request,
            "handle_system_information_request_node",
        ),
    )
    builder.add_node(
        "converse_with_bt_servant_node",
        wrap_node_with_timing(
            brain_nodes.converse_with_bt_servant, "converse_with_bt_servant_node"
        ),
    )
    builder.add_node(
        "handle_get_passage_summary_node",
        wrap_node_with_progress(
            brain_nodes.handle_get_passage_summary,
            "handle_get_passage_summary_node",
            progress_message="I'm gathering the passage details so I can summarize them for you.",
            force=True,
        ),
    )
    builder.add_node(
        "handle_get_passage_keywords_node",
        wrap_node_with_progress(
            brain_nodes.handle_get_passage_keywords,
            "handle_get_passage_keywords_node",
            progress_message="I'm reviewing the passage to pull out its key words for you.",
            force=True,
        ),
    )
    builder.add_node(
        "handle_get_translation_helps_node",
        wrap_node_with_progress(
            brain_nodes.handle_get_translation_helps,
            "handle_get_translation_helps_node",
            progress_message="I'm compiling translation helps that will support your work on this passage.",
            force=True,
        ),
    )
    builder.add_node(
        "handle_retrieve_scripture_node",
        wrap_node_with_timing(
            brain_nodes.handle_retrieve_scripture, "handle_retrieve_scripture_node"
        ),
    )
    builder.add_node(
        "handle_listen_to_scripture_node",
        wrap_node_with_timing(
            brain_nodes.handle_listen_to_scripture, "handle_listen_to_scripture_node"
        ),
    )
    builder.add_node(
        "handle_translate_scripture_node",
        wrap_node_with_progress(
            brain_nodes.handle_translate_scripture,
            "handle_translate_scripture_node",
            progress_message="I'm translating this passage into the language you asked for.",
            force=True,
        ),
    )
    builder.add_node(
        "translate_responses_node",
        wrap_node_with_progress(
            brain_nodes.translate_responses,
            "translate_responses_node",
            progress_message="I'm translating my response into your preferred language now.",
            condition=_should_show_translation_progress,
            force=True,
        ),
        defer=True,
    )

    # Set up edges
    builder.set_entry_point("start_node")
    builder.add_edge("start_node", "determine_query_language_node")
    builder.add_edge("determine_query_language_node", "preprocess_user_query_node")
    builder.add_edge("preprocess_user_query_node", "determine_intents_node")
    builder.add_conditional_edges("determine_intents_node", process_intents)
    builder.add_edge("query_vector_db_node", "query_open_ai_node")
    builder.add_edge("set_response_language_node", "translate_responses_node")
    builder.add_edge("set_agentic_strength_node", "translate_responses_node")
    # After chunking, finish. Do not loop back to translate, which can recreate
    # the long message and trigger an infinite chunk cycle.

    builder.add_edge("handle_unsupported_function_node", "translate_responses_node")
    builder.add_edge("handle_system_information_request_node", "translate_responses_node")
    builder.add_edge("converse_with_bt_servant_node", "translate_responses_node")
    builder.add_edge("handle_get_passage_summary_node", "translate_responses_node")
    builder.add_edge("handle_get_passage_keywords_node", "translate_responses_node")
    builder.add_edge("handle_get_translation_helps_node", "translate_responses_node")
    builder.add_edge("handle_retrieve_scripture_node", "translate_responses_node")
    builder.add_edge("handle_listen_to_scripture_node", "translate_responses_node")
    builder.add_edge("handle_translate_scripture_node", "translate_responses_node")
    builder.add_edge("query_open_ai_node", "translate_responses_node")
    builder.add_edge("consult_fia_resources_node", "translate_responses_node")

    builder.add_conditional_edges("translate_responses_node", brain_nodes.needs_chunking)
    builder.set_finish_point("chunk_message_node")

    return builder.compile()


__all__ = [
    "BrainState",
    "create_brain",
    "process_intents",
    "wrap_node_with_timing",
    "wrap_node_with_progress",
]
