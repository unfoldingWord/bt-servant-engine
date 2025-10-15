"""LangGraph orchestration setup for BT Servant brain.

This module contains the graph construction, node routing, and timing infrastructure
for the brain decision pipeline.
"""

from __future__ import annotations

import operator
import re
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
    progress_messenger: Optional[Callable[[str], Awaitable[None]]]
    last_progress_time: float
    progress_throttle_seconds: float
    # Intent queue fields for sequential multi-intent processing
    intents_with_context: Optional[List[Any]]  # IntentWithContext list from structured detection
    queued_intent_context: Optional[Dict[str, Any]]  # Parameters from queued intent
    has_more_queued_intents: bool  # Whether user has remaining queued intents
    next_queued_intent_preview: Optional[str]  # Preview of next intent for continuation prompt
    active_intent: IntentType
    active_intent_query: str
    suppress_internal_followups: bool
    deferred_intent_topics: List[str]


ProgressMessageInput = str | Callable[[Any], Optional[str]]


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


def _normalize_followup_prompt(prompt: Optional[str]) -> Optional[str]:
    if not prompt:
        return None
    text = prompt.strip()
    if not text:
        return None
    text = re.sub(r"[?]+$", "", text).strip()
    lowered = text.lower()
    prefixes = [
        "would you like me to",
        "would you like to",
        "do you want me to",
        "should i",
        "could i",
        "can i",
        "shall i",
    ]
    for prefix in prefixes:
        if lowered.startswith(prefix):
            text = text[len(prefix) :].lstrip(",. ").strip()
            break
    if not text:
        return None
    # Capitalize first letter for a clean imperative-style command
    return text[0].upper() + text[1:]


def _build_intent_query(
    intent: IntentType,
    base_query: str,
    params: Optional[Dict[str, Any]],
    fallback_prompt: Optional[str],
) -> str:
    """Derive a focused query string for the active intent."""

    def _get_str(key: str) -> Optional[str]:
        value = (params or {}).get(key)
        if isinstance(value, str):
            stripped = value.strip()
            return stripped or None
        return None

    intent_query: Optional[str] = None

    if intent == IntentType.GET_BIBLE_TRANSLATION_ASSISTANCE:
        intent_query = _get_str("query") or _get_str("topic") or _get_str("subject")
    elif intent == IntentType.CONSULT_FIA_RESOURCES:
        focus = _get_str("topic") or _get_str("focus") or _get_str("passage")
        if focus:
            intent_query = f"Explain the FIA process for {focus}"
    elif intent == IntentType.GET_PASSAGE_SUMMARY:
        passage = _get_str("passage")
        if passage:
            intent_query = f"Summarize {passage}"
    elif intent == IntentType.GET_PASSAGE_KEYWORDS:
        passage = _get_str("passage")
        if passage:
            intent_query = f"List key terms for {passage}"
    elif intent == IntentType.GET_TRANSLATION_HELPS:
        passage = _get_str("passage")
        if passage:
            intent_query = f"Provide translation helps for {passage}"
    elif intent == IntentType.RETRIEVE_SCRIPTURE:
        passage = _get_str("passage")
        if passage:
            intent_query = f"Retrieve scripture for {passage}"
    elif intent == IntentType.LISTEN_TO_SCRIPTURE:
        passage = _get_str("passage")
        if passage:
            intent_query = f"Read aloud {passage}"
    elif intent == IntentType.TRANSLATE_SCRIPTURE:
        passage = _get_str("passage")
        target = _get_str("target_language") or _get_str("targetLanguage")
        if passage and target:
            intent_query = f"Translate {passage} to {target}"
        elif passage:
            intent_query = f"Translate {passage}"

    if not intent_query:
        intent_query = _normalize_followup_prompt(fallback_prompt)

    final_query = (intent_query or base_query).strip()
    return final_query or base_query


def build_translation_assistance_progress_message(state: Any) -> Optional[str]:
    """Generate the progress message for translation assistance summarizing resource origins."""
    s = cast(BrainState, state)
    docs = cast(List[Dict[str, Any]], s.get("docs", []))
    sources = _collect_resource_sources(docs)
    if not sources:
        return status_messages.get_status_message(status_messages.FINALIZING_RESPONSE, s)

    resources_list = _format_series(sources)
    found_docs_msg = status_messages.get_status_message(status_messages.FOUND_RELEVANT_DOCUMENTS, s)
    finalizing_msg = status_messages.get_status_message(status_messages.FINALIZING_RESPONSE, s)
    return f"{found_docs_msg.replace('{resources}', resources_list)} {finalizing_msg}"


def process_intents(state: Any) -> List[Hashable]:  # pylint: disable=too-many-branches
    """Map detected intents to the list of nodes to traverse.

    Sequential processing: Only handles the highest priority intent and queues the rest.
    When multiple intents are detected, we process the highest priority one and save
    the others to the intent queue for subsequent requests.
    """
    import time

    from bt_servant_engine.core.intents import IntentQueueItem
    from bt_servant_engine.services.intent_queue import (
        has_queued_intents,
        pop_next_intent,
        save_intent_queue,
    )

    s = cast(BrainState, state)
    user_intents = s["user_intents"]
    user_id = s["user_id"]

    if not user_intents:
        raise ValueError("no intents found. something went very wrong.")

    intents_to_queue: List[IntentType] = []
    deferred_topics: List[str] = []
    # Check if we're processing a queued intent
    queued_intent_context = s.get("queued_intent_context")
    if queued_intent_context:
        # Pop from queue since we're processing it
        logger.info(
            "[process-intents] Processing queued intent, popping from queue for user=%s", user_id
        )
        pop_next_intent(user_id)

    # Determine which intent to process
    multi_intent = len(user_intents) > 1
    if not multi_intent:
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

        # Get structured context if available
        intents_with_context = s.get("intents_with_context")
        continuation_actions = cast(list[str], s.get("continuation_actions", []))

        # Get original intent order (before sorting) for action lookup
        original_intent_order = (
            [ic.intent for ic in intents_with_context] if intents_with_context else user_intents
        )

        # Build queue items
        queue_items: list[IntentQueueItem] = []
        for intent in intents_to_queue:
            # Find matching context if available
            params = {}
            if intents_with_context:
                matching_context = next(
                    (ic for ic in intents_with_context if ic.intent == intent), None
                )
                if matching_context and matching_context.parameters:
                    params = matching_context.parameters

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

            queue_items.append(
                IntentQueueItem(
                    intent=intent,
                    parameters=params,
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

        for item in queue_items:
            topic = ""
            if isinstance(item.parameters, dict):
                topic = (
                    cast(str, item.parameters.get("query", ""))
                    or cast(str, item.parameters.get("topic", ""))
                    or cast(str, item.parameters.get("passage", ""))
                ).strip()
            if not topic and item.continuation_action:
                topic = _normalize_followup_prompt(item.continuation_action) or ""
            if topic:
                deferred_topics.append(topic)

    intents_with_context = s.get("intents_with_context")
    continuation_actions = cast(list[str], s.get("continuation_actions", []))
    original_intent_order = (
        [ic.intent for ic in intents_with_context] if intents_with_context else user_intents
    )

    # Determine parameters for the active intent
    active_params: Dict[str, Any] = {}
    if queued_intent_context:
        active_params = queued_intent_context
    elif intents_with_context:
        matching_context = next(
            (ic for ic in intents_with_context if ic.intent == intent_to_process), None
        )
        if matching_context and matching_context.parameters:
            active_params = matching_context.parameters

    # Continuation action (if generated) in original detection order
    action_prompt: Optional[str] = None
    try:
        action_idx = original_intent_order.index(intent_to_process)
    except ValueError:
        action_idx = -1
    if action_idx >= 0 and action_idx < len(continuation_actions):
        action_prompt = continuation_actions[action_idx]

    base_query = s.get("transformed_query", "")
    active_intent_query = _build_intent_query(intent_to_process, base_query, active_params, action_prompt)

    s["active_intent"] = intent_to_process
    s["active_intent_query"] = active_intent_query

    if multi_intent:
        suppress_internal_followups = True
        s["has_more_queued_intents"] = bool(intents_to_queue)
    elif queued_intent_context:
        suppress_internal_followups = has_queued_intents(user_id)
        s["has_more_queued_intents"] = suppress_internal_followups
    else:
        suppress_internal_followups = False
        s["has_more_queued_intents"] = False
    s["suppress_internal_followups"] = suppress_internal_followups
    s["deferred_intent_topics"] = deferred_topics

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
            progress_message=lambda s: status_messages.get_status_message(
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
            progress_message=lambda s: status_messages.get_status_message(
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
            progress_message=lambda s: status_messages.get_status_message(
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
            progress_message=lambda s: status_messages.get_status_message(
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
            progress_message=lambda s: status_messages.get_status_message(
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
            progress_message=lambda s: status_messages.get_status_message(
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
            progress_message=lambda s: status_messages.get_status_message(
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
            progress_message=lambda s: status_messages.get_status_message(
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
            progress_message=lambda s: status_messages.get_status_message(
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
            progress_message=lambda s: status_messages.get_status_message(
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
            progress_message=lambda s: status_messages.get_status_message(
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
            progress_message=lambda s: status_messages.get_status_message(
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
