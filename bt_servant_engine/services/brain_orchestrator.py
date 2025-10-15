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
from bt_servant_engine.core.logging import get_logger
from bt_servant_engine.services import status_messages
from utils.perf import set_current_trace, time_block

logger = get_logger(__name__)

if TYPE_CHECKING:
    pass


# Intent priority map for sequential processing
# Higher values = higher priority = processed first when multiple intents detected
INTENT_PRIORITY: Dict[IntentType, int] = {
    # Settings intents: Always process first to configure the session
    IntentType.SET_RESPONSE_LANGUAGE: 100,
    IntentType.SET_AGENTIC_STRENGTH: 99,
    # Scripture retrieval: Get the text before analyzing it
    IntentType.RETRIEVE_SCRIPTURE: 80,
    IntentType.LISTEN_TO_SCRIPTURE: 79,  # Audio variant of retrieval
    # Analytical intents: Process after retrieval if both present
    IntentType.GET_PASSAGE_SUMMARY: 70,
    IntentType.GET_PASSAGE_KEYWORDS: 69,
    IntentType.GET_TRANSLATION_HELPS: 68,
    # Translation: After analysis
    IntentType.TRANSLATE_SCRIPTURE: 65,
    # FIA and general assistance
    IntentType.CONSULT_FIA_RESOURCES: 50,
    IntentType.GET_BIBLE_TRANSLATION_ASSISTANCE: 45,
    # Conversational and utility intents
    IntentType.CONVERSE_WITH_BT_SERVANT: 20,
    # System and unsupported: Lowest priority
    IntentType.RETRIEVE_SYSTEM_INFORMATION: 10,
    IntentType.PERFORM_UNSUPPORTED_FUNCTION: 5,
}


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
    progress_messenger: Optional[
        Callable[[status_messages.LocalizedProgressMessage], Awaitable[None]]
    ]
    last_progress_time: float
    progress_throttle_seconds: float
    # Intent queue fields for sequential multi-intent processing
    intents_with_context: Optional[List[Any]]  # IntentWithContext list from structured detection
    continuation_actions: Optional[List[str]]
    queued_intent_context: Optional[str]  # Context text from queued intent
    intent_context_map: Optional[Dict[str, str]]  # Mapping of intent value to context snippet
    active_intent_context: str  # Effective query text for the node currently executing
    active_intent_context_source: str  # Origin of the active context (structured, queue, fallback)
    # Follow-up question tracking - prevents duplicate follow-ups when multi-intent is active
    followup_question_added: bool  # Set to True when any follow-up question is added to response


ProgressMessageInput = (
    status_messages.LocalizedProgressMessage
    | str
    | Callable[[Any], Optional[status_messages.LocalizedProgressMessage | str]]
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
        message_payload: Optional[status_messages.LocalizedProgressMessage] = None
        if progress_message is not None and should_show:
            raw_message = (
                progress_message(state) if callable(progress_message) else progress_message
            )
            if isinstance(raw_message, str):
                message_payload = status_messages.make_progress_message(raw_message)
            elif isinstance(raw_message, dict):
                message_payload = cast(status_messages.LocalizedProgressMessage, raw_message)
            else:
                message_payload = raw_message

        if message_payload:
            # Import here to avoid circular dependency
            from bt_servant_engine.services.progress_messaging import maybe_send_progress

            coroutine = maybe_send_progress(state, message_payload, force=force)
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
        ordered.append(normalized.replace("_", " "))
    return ordered


def build_translation_assistance_progress_message(
    state: Any,
) -> Optional[status_messages.LocalizedProgressMessage]:
    """Generate the progress message for translation assistance summarizing resource origins."""
    s = cast(BrainState, state)
    docs = cast(List[Dict[str, Any]], s.get("docs", []))
    sources = _collect_resource_sources(docs)
    if not sources:
        return status_messages.get_progress_message(status_messages.FINALIZING_RESPONSE, s)

    resources_list = _format_series(sources)
    found_docs_msg = status_messages.get_status_message(
        status_messages.FOUND_RELEVANT_DOCUMENTS, s, resources=resources_list
    )
    finalizing_msg = status_messages.get_status_message(status_messages.FINALIZING_RESPONSE, s)
    combined = f"{found_docs_msg} {finalizing_msg}"
    return status_messages.make_progress_message(
        combined,
        message_key=status_messages.FOUND_RELEVANT_DOCUMENTS,
    )


def process_intents(state: Any) -> List[Hashable]:  # pylint: disable=too-many-branches
    """Map detected intents to the list of nodes to traverse.

    Sequential processing: Only handles the highest priority intent and queues the rest.
    When multiple intents are detected, we process the highest priority one and save
    the others to the intent queue for subsequent requests.
    """
    import time

    from bt_servant_engine.core.intents import IntentQueueItem
    from bt_servant_engine.services.intent_queue import pop_next_intent, save_intent_queue

    s = cast(BrainState, state)
    user_intents = s["user_intents"]
    user_id = s["user_id"]

    if not user_intents:
        raise ValueError("no intents found. something went very wrong.")

    intents_with_context = cast(list[Any] | None, s.get("intents_with_context"))
    continuation_actions = cast(list[str], s.get("continuation_actions", []))
    intent_context_map = cast(dict[str, str], s.get("intent_context_map") or {})
    context_pool = list(intents_with_context or [])
    active_context = ""
    active_context_source = "transformed_query"

    # Check if we're processing a queued intent
    queued_intent_context = s.get("queued_intent_context")
    if queued_intent_context:
        # Pop from queue since we're processing it
        logger.info(
            "[process-intents] Processing queued intent, popping from queue for user=%s", user_id
        )
        pop_next_intent(user_id)
        active_context = str(queued_intent_context).strip()
        active_context_source = "queued_intent_context"
        logger.info(
            "[process-intents] Using queued intent context for user=%s: '%s'",
            user_id,
            active_context,
        )
        # Clear the queued context from state to avoid re-use in later turns
        s["queued_intent_context"] = None
        if user_intents:
            user_intents[:] = [user_intents[0]]

    # Determine which intent to process
    if len(user_intents) == 1:
        # Single intent - process it
        intent_to_process = user_intents[0]
        logger.info(
            "[process-intents] Single intent: %s for user=%s", intent_to_process.value, user_id
        )
    else:
        # Multiple intents - process highest priority, queue the rest
        logger.info(
            "[process-intents] Multiple intents detected (%d), sorting by priority for user=%s",
            len(user_intents),
            user_id,
        )

        # Sort by priority (highest first)
        sorted_intents = sorted(user_intents, key=lambda i: INTENT_PRIORITY.get(i, 0), reverse=True)
        intent_to_process = sorted_intents[0]
        intents_to_queue = sorted_intents[1:]

        logger.info(
            "[process-intents] Processing highest priority intent: %s (priority=%d) for user=%s",
            intent_to_process.value,
            INTENT_PRIORITY.get(intent_to_process, 0),
            user_id,
        )
        # Trim user_intents down to the active intent for downstream nodes
        user_intents[:] = [intent_to_process]

        # Determine context snippet for the active intent if not set via queue
        if not active_context and context_pool:
            active_idx = next(
                (idx for idx, ic in enumerate(context_pool) if ic.intent == intent_to_process),
                None,
            )
            active_match = context_pool.pop(active_idx) if active_idx is not None else None
            if active_match:
                active_context = active_match.trimmed_context()
                active_context_source = "structured_context"
                logger.info(
                    "[process-intents] Active intent context (structured) for user=%s: '%s'",
                    user_id,
                    active_context,
                )
            else:
                logger.warning(
                    "[process-intents] Structured context missing for active intent=%s (user=%s); will rely on fallback context",
                    intent_to_process.value,
                    user_id,
                )
        if not active_context:
            mapped_context = intent_context_map.get(intent_to_process.value, "")
            if mapped_context:
                active_context = mapped_context
                active_context_source = "intent_context_map"
                logger.info(
                    "[process-intents] Using mapped context for active intent %s: '%s'",
                    intent_to_process.value,
                    active_context,
                )

        # Get original intent order (before sorting) for action lookup
        original_intent_order = (
            [ic.intent for ic in intents_with_context] if intents_with_context else sorted_intents
        )

        # Build queue items
        queue_items = []
        for intent in intents_to_queue:
            # Find matching context snippet if available
            context_text = ""
            if context_pool:
                matching_index = next(
                    (idx for idx, ic in enumerate(context_pool) if ic.intent == intent),
                    None,
                )
                matching_context = (
                    context_pool.pop(matching_index) if matching_index is not None else None
                )
                if matching_context:
                    context_text = matching_context.trimmed_context()
            if not context_text:
                context_text = intent_context_map.get(intent.value, "")
            if not context_text:
                logger.warning(
                    "[process-intents] No context snippet available for deferred intent=%s (user=%s); downstream will fall back to active context resolution",
                    intent.value,
                    user_id,
                )

            # Get corresponding continuation action using ORIGINAL order (not sorted)
            # continuation_actions are in the same order as original intent detection
            try:
                action_idx = original_intent_order.index(intent)
                continuation_action = (
                    continuation_actions[action_idx]
                    if action_idx < len(continuation_actions)
                    else ""
                )
            except ValueError:
                # Intent not found in original order (shouldn't happen)
                continuation_action = ""
                logger.warning(
                    "[process-intents] Intent %s not found in original order for user=%s",
                    intent.value,
                    user_id,
                )

            logger.info(
                "[process-intents] Queueing deferred intent=%s with context='%s' (user=%s, continuation_action_present=%s)",
                intent.value,
                context_text,
                user_id,
                bool(continuation_action),
            )

            queue_items.append(
                IntentQueueItem(
                    intent=intent,
                    context_text=context_text,
                    continuation_action=continuation_action,
                    created_at=time.time(),
                    original_query=s["user_query"],
                )
            )

        # Save queue
        logger.info(
            "[process-intents] Queueing %d remaining intents for user=%s", len(queue_items), user_id
        )
        save_intent_queue(user_id, queue_items)

    # Fallback: if no explicit context captured, default to transformed query
    if not active_context:
        active_context = s["transformed_query"]
        active_context_source = "transformed_query"
        logger.info(
            "[process-intents] No specialized context found; defaulting to transformed query for user=%s: '%s'",
            user_id,
            active_context,
        )

    s["active_intent_context"] = active_context
    s["active_intent_context_source"] = active_context_source
    logger.info(
        "[process-intents] Intent routing context established (source=%s, intent=%s, user=%s)",
        active_context_source,
        intent_to_process.value,
        user_id,
    )

    # Map intent to node (using elif to avoid duplicates from old code)
    nodes_to_traverse: List[Hashable] = []

    if intent_to_process == IntentType.GET_BIBLE_TRANSLATION_ASSISTANCE:
        nodes_to_traverse.append("query_vector_db_node")
    elif intent_to_process == IntentType.CONSULT_FIA_RESOURCES:
        nodes_to_traverse.append("consult_fia_resources_node")
    elif intent_to_process == IntentType.GET_PASSAGE_SUMMARY:
        nodes_to_traverse.append("handle_get_passage_summary_node")
    elif intent_to_process == IntentType.GET_PASSAGE_KEYWORDS:
        nodes_to_traverse.append("handle_get_passage_keywords_node")
    elif intent_to_process == IntentType.GET_TRANSLATION_HELPS:
        nodes_to_traverse.append("handle_get_translation_helps_node")
    elif intent_to_process == IntentType.RETRIEVE_SCRIPTURE:
        nodes_to_traverse.append("handle_retrieve_scripture_node")
    elif intent_to_process == IntentType.LISTEN_TO_SCRIPTURE:
        nodes_to_traverse.append("handle_listen_to_scripture_node")
    elif intent_to_process == IntentType.SET_RESPONSE_LANGUAGE:
        nodes_to_traverse.append("set_response_language_node")
    elif intent_to_process == IntentType.SET_AGENTIC_STRENGTH:
        nodes_to_traverse.append("set_agentic_strength_node")
    elif intent_to_process == IntentType.PERFORM_UNSUPPORTED_FUNCTION:
        nodes_to_traverse.append("handle_unsupported_function_node")
    elif intent_to_process == IntentType.RETRIEVE_SYSTEM_INFORMATION:
        nodes_to_traverse.append("handle_system_information_request_node")
    elif intent_to_process == IntentType.CONVERSE_WITH_BT_SERVANT:
        nodes_to_traverse.append("converse_with_bt_servant_node")
    elif intent_to_process == IntentType.TRANSLATE_SCRIPTURE:
        nodes_to_traverse.append("handle_translate_scripture_node")
    else:
        raise ValueError(f"Unknown intent type: {intent_to_process}")

    logger.info("[process-intents] Routing to node: %s for user=%s", nodes_to_traverse[0], user_id)

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
            progress_message=lambda s: status_messages.get_progress_message(
                status_messages.THINKING_ABOUT_MESSAGE, s
            ),
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
            progress_message=lambda s: status_messages.get_progress_message(
                status_messages.SEARCHING_BIBLE_RESOURCES, s
            ),
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
        wrap_node_with_progress(
            brain_nodes.consult_fia_resources,
            "consult_fia_resources_node",
            progress_message=lambda s: status_messages.get_progress_message(
                status_messages.REVIEWING_FIA_GUIDANCE, s
            ),
            force=True,
        ),
    )
    builder.add_node(
        "chunk_message_node", wrap_node_with_timing(brain_nodes.chunk_message, "chunk_message_node")
    )
    builder.add_node(
        "handle_unsupported_function_node",
        wrap_node_with_progress(
            brain_nodes.handle_unsupported_function,
            "handle_unsupported_function_node",
            progress_message=lambda s: status_messages.get_progress_message(
                status_messages.CHECKING_CAPABILITIES, s
            ),
            force=True,
        ),
    )
    builder.add_node(
        "handle_system_information_request_node",
        wrap_node_with_progress(
            brain_nodes.handle_system_information_request,
            "handle_system_information_request_node",
            progress_message=lambda s: status_messages.get_progress_message(
                status_messages.GENERATING_HELP_RESPONSE, s
            ),
            force=True,
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
            progress_message=lambda s: status_messages.get_progress_message(
                status_messages.GATHERING_PASSAGE_SUMMARY, s
            ),
            force=True,
        ),
    )
    builder.add_node(
        "handle_get_passage_keywords_node",
        wrap_node_with_progress(
            brain_nodes.handle_get_passage_keywords,
            "handle_get_passage_keywords_node",
            progress_message=lambda s: status_messages.get_progress_message(
                status_messages.EXTRACTING_KEYWORDS, s
            ),
            force=True,
        ),
    )
    builder.add_node(
        "handle_get_translation_helps_node",
        wrap_node_with_progress(
            brain_nodes.handle_get_translation_helps,
            "handle_get_translation_helps_node",
            progress_message=lambda s: status_messages.get_progress_message(
                status_messages.COMPILING_TRANSLATION_HELPS, s
            ),
            force=True,
        ),
    )
    builder.add_node(
        "handle_retrieve_scripture_node",
        wrap_node_with_progress(
            brain_nodes.handle_retrieve_scripture,
            "handle_retrieve_scripture_node",
            progress_message=lambda s: status_messages.get_progress_message(
                status_messages.GATHERING_PASSAGE_TEXT, s
            ),
            force=True,
        ),
    )
    builder.add_node(
        "handle_listen_to_scripture_node",
        wrap_node_with_progress(
            brain_nodes.handle_listen_to_scripture,
            "handle_listen_to_scripture_node",
            progress_message=lambda s: status_messages.get_progress_message(
                status_messages.PREPARING_AUDIO, s
            ),
            force=True,
        ),
    )
    builder.add_node(
        "handle_translate_scripture_node",
        wrap_node_with_progress(
            brain_nodes.handle_translate_scripture,
            "handle_translate_scripture_node",
            progress_message=lambda s: status_messages.get_progress_message(
                status_messages.TRANSLATING_PASSAGE, s
            ),
            force=True,
        ),
    )
    builder.add_node(
        "translate_responses_node",
        wrap_node_with_progress(
            brain_nodes.translate_responses,
            "translate_responses_node",
            progress_message=lambda s: status_messages.get_progress_message(
                status_messages.TRANSLATING_RESPONSE, s
            ),
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
