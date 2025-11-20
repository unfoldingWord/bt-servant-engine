"""LangGraph orchestration setup for BT Servant brain.

This module contains the graph construction, node routing, and timing infrastructure
for the brain decision pipeline.
"""

from __future__ import annotations

import asyncio
import operator
import time
from dataclasses import dataclass
from collections.abc import Hashable
from typing import Annotated, Any, Awaitable, Callable, Dict, List, Optional, cast

from langgraph.graph import StateGraph
from langgraph.types import Send
from typing_extensions import TypedDict

from bt_servant_engine.core.config import config
from bt_servant_engine.core.intents import IntentQueueItem, IntentType
from bt_servant_engine.core.logging import get_logger
from bt_servant_engine.services import brain_nodes, status_messages
from bt_servant_engine.services.intent_queue import pop_next_intent, save_intent_queue
from bt_servant_engine.services.progress_messaging import (
    maybe_send_progress,
    should_show_translation_progress,
)
from utils.perf import set_current_trace, time_block
from utils.identifiers import get_log_safe_user_id

logger = get_logger(__name__)

# Intent priority map for sequential processing
# Higher values = higher priority = processed first when multiple intents detected
INTENT_PRIORITY: Dict[IntentType, int] = {
    # Settings intents: Always process first to configure the session
    IntentType.CLEAR_RESPONSE_LANGUAGE: 101,
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

INTENT_NODE_MAP: Dict[IntentType, str] = {
    IntentType.GET_BIBLE_TRANSLATION_ASSISTANCE: "query_vector_db_node",
    IntentType.CONSULT_FIA_RESOURCES: "consult_fia_resources_node",
    IntentType.GET_PASSAGE_SUMMARY: "handle_get_passage_summary_node",
    IntentType.GET_PASSAGE_KEYWORDS: "handle_get_passage_keywords_node",
    IntentType.GET_TRANSLATION_HELPS: "handle_get_translation_helps_node",
    IntentType.RETRIEVE_SCRIPTURE: "handle_retrieve_scripture_node",
    IntentType.LISTEN_TO_SCRIPTURE: "handle_listen_to_scripture_node",
    IntentType.SET_RESPONSE_LANGUAGE: "set_response_language_node",
    IntentType.CLEAR_RESPONSE_LANGUAGE: "clear_response_language_node",
    IntentType.SET_AGENTIC_STRENGTH: "set_agentic_strength_node",
    IntentType.PERFORM_UNSUPPORTED_FUNCTION: "handle_unsupported_function_node",
    IntentType.RETRIEVE_SYSTEM_INFORMATION: "handle_system_information_request_node",
    IntentType.CONVERSE_WITH_BT_SERVANT: "converse_with_bt_servant_node",
    IntentType.TRANSLATE_SCRIPTURE: "handle_translate_scripture_node",
}

_SINGLE_RESOURCE_COUNT = 1
_PAIR_RESOURCE_COUNT = 2


def _log_safe_user(user_id: str) -> str:
    return get_log_safe_user_id(user_id, secret=config.LOG_PSEUDONYM_SECRET)


@dataclass(slots=True)
class ActiveContext:
    """Describes the text and provenance for the intent context."""

    text: str = ""
    source: str = "transformed_query"


@dataclass(slots=True)
class IntentSelection:
    """Primary intent to process plus any deferred intents to queue."""

    primary: IntentType
    deferred: List[IntentType]


@dataclass(slots=True)
class DeferredQueueContext:
    """Inputs required to build intent queue entries."""

    selection: IntentSelection
    context_pool: List[Any]
    intent_context_map: Dict[str, str]
    continuation_actions: List[str]
    intents_with_context: Optional[List[Any]]
    user_id: str
    original_query: str


def _consume_queued_intent(
    state: BrainState,
    user_id: str,
    updates: Dict[str, Any],
    pop_next: Callable[[str], Optional[IntentQueueItem]],
) -> ActiveContext:
    queued_context = state.get("queued_intent_context")
    if not queued_context:
        return ActiveContext()

    log_user_id = _log_safe_user(user_id)
    logger.info(
        "[process-intents] Processing queued intent, popping from queue for user=%s", log_user_id
    )
    pop_next(user_id)
    context_text = str(queued_context).strip()
    logger.info(
        "[process-intents] Using queued intent context for user=%s: '%s'",
        log_user_id,
        context_text,
    )
    updates["queued_intent_context"] = None
    if state.get("user_intents"):
        updates["user_intents"] = [state["user_intents"][0]]
    return ActiveContext(text=context_text, source="queued_intent_context")


def _select_intent(user_intents: List[IntentType], user_id: str) -> IntentSelection:
    log_user_id = _log_safe_user(user_id)
    if len(user_intents) == 1:
        intent = user_intents[0]
        logger.info("[process-intents] Single intent: %s for user=%s", intent.value, log_user_id)
        return IntentSelection(primary=intent, deferred=[])

    logger.info(
        "[process-intents] Multiple intents detected (%d), sorting by priority for user=%s",
        len(user_intents),
        log_user_id,
    )
    sorted_intents = sorted(user_intents, key=lambda i: INTENT_PRIORITY.get(i, 0), reverse=True)
    primary = sorted_intents[0]
    deferred = sorted_intents[1:]
    logger.info(
        "[process-intents] Processing highest priority intent: %s (priority=%d) for user=%s",
        primary.value,
        INTENT_PRIORITY.get(primary, 0),
        log_user_id,
    )
    return IntentSelection(primary=primary, deferred=deferred)


def _resolve_active_context(
    current: ActiveContext,
    *,
    context_pool: List[Any],
    intent_context_map: Dict[str, str],
    primary_intent: IntentType,
    user_id: str,
) -> ActiveContext:
    log_user_id = _log_safe_user(user_id)
    if current.text:
        return current

    if context_pool:
        index = next(
            (idx for idx, item in enumerate(context_pool) if item.intent == primary_intent),
            None,
        )
        match = context_pool.pop(index) if index is not None else None
        if match:
            context_text = match.trimmed_context()
            logger.info(
                "[process-intents] Active intent context (structured) for user=%s: '%s'",
                log_user_id,
                context_text,
            )
            return ActiveContext(text=context_text, source="structured_context")
        logger.warning(
            "[process-intents] Structured context missing for intent=%s (user=%s); falling back",
            primary_intent.value,
            log_user_id,
        )

    mapped_context = intent_context_map.get(primary_intent.value, "")
    if mapped_context:
        logger.info(
            "[process-intents] Using mapped context for active intent %s: '%s'",
            primary_intent.value,
            mapped_context,
        )
        return ActiveContext(text=mapped_context, source="intent_context_map")
    return current


def _queue_deferred_intents(queue_ctx: DeferredQueueContext) -> None:
    if not queue_ctx.selection.deferred:
        return

    log_user_id = _log_safe_user(queue_ctx.user_id)

    original_order = (
        [ic.intent for ic in queue_ctx.intents_with_context]
        if queue_ctx.intents_with_context
        else [queue_ctx.selection.primary, *queue_ctx.selection.deferred]
    )

    queue_items: List[IntentQueueItem] = []
    for intent in queue_ctx.selection.deferred:
        context_text = ""
        if queue_ctx.context_pool:
            idx = next(
                (i for i, item in enumerate(queue_ctx.context_pool) if item.intent == intent),
                None,
            )
            match = queue_ctx.context_pool.pop(idx) if idx is not None else None
            if match:
                context_text = match.trimmed_context()
        if not context_text:
            context_text = queue_ctx.intent_context_map.get(intent.value, "")
        if not context_text:
            logger.warning(
                "[process-intents] Missing context for deferred intent=%s "
                "(user=%s); using fallback",
                intent.value,
                log_user_id,
            )

        try:
            action_idx = original_order.index(intent)
            continuation_action = (
                queue_ctx.continuation_actions[action_idx]
                if action_idx < len(queue_ctx.continuation_actions)
                else ""
            )
        except ValueError:
            continuation_action = ""
            logger.warning(
                "[process-intents] Intent %s not found in original order for user=%s",
                intent.value,
                log_user_id,
            )

        logger.info(
            "[process-intents] Queueing deferred intent=%s with context='%s' "
            "(user=%s, continuation_action=%s)",
            intent.value,
            context_text,
            log_user_id,
            bool(continuation_action),
        )
        queue_items.append(
            IntentQueueItem(
                intent=intent,
                context_text=context_text,
                continuation_action=continuation_action,
                created_at=time.time(),
                original_query=queue_ctx.original_query,
            )
        )

    logger.info(
        "[process-intents] Queueing %d remaining intents for user=%s",
        len(queue_items),
        log_user_id,
    )
    save_intent_queue(queue_ctx.user_id, queue_items)


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
    passage_followup_context: Dict[str, Any]


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


_SERIES_JOINERS: Dict[str, Dict[str, str]] = {
    # pair: separator for two items
    # middle: separator between items except the last
    # final: separator before the last item in lists with 3+ entries
    "en": {"pair": " and ", "middle": ", ", "final": ", and "},
    "ar": {"pair": " \u0648 ", "middle": "\u060c ", "final": " \u0648 "},
    "fr": {"pair": " et ", "middle": ", ", "final": ", et "},
    "es": {"pair": " y ", "middle": ", ", "final": ", y "},
    "pt": {"pair": " e ", "middle": ", ", "final": ", e "},
    "ru": {"pair": " \u0438 ", "middle": ", ", "final": ", \u0438 "},
    "id": {"pair": " dan ", "middle": ", ", "final": ", dan "},
    "sw": {"pair": " na ", "middle": ", ", "final": ", na "},
    "nl": {"pair": " en ", "middle": ", ", "final": ", en "},
    "hi": {"pair": " \u0914\u0930 ", "middle": ", ", "final": ", \u0914\u0930 "},
    "zh": {"pair": "\u548c", "middle": "\u3001", "final": "\u548c"},
}


def _normalized_language_code(language: str) -> str:
    return language.split("-", maxsplit=1)[0].strip().lower() or "en"


def _format_series(resources: List[str], language: str) -> str:
    if not resources:
        return ""
    joiners = _SERIES_JOINERS.get(_normalized_language_code(language), _SERIES_JOINERS["en"])
    if len(resources) == _SINGLE_RESOURCE_COUNT:
        return resources[0]
    if len(resources) == _PAIR_RESOURCE_COUNT:
        return f"{resources[0]}{joiners['pair']}{resources[1]}"
    return f"{joiners['middle'].join(resources[:-1])}{joiners['final']}{resources[-1]}"


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

    target_language = status_messages.get_effective_response_language(s)
    resources_list = _format_series(sources, target_language)
    found_docs_payload = status_messages.get_progress_message(
        status_messages.FOUND_RELEVANT_DOCUMENTS, s, resources=resources_list
    )
    finalizing_msg = status_messages.get_status_message(status_messages.FINALIZING_RESPONSE, s)
    combined = f"{found_docs_payload['text']} {finalizing_msg}"
    return status_messages.make_progress_message(
        combined,
        message_key=status_messages.FOUND_RELEVANT_DOCUMENTS,
    )


def process_intents(state: Any) -> List[Hashable]:
    """Map detected intents to the list of nodes to traverse.

    Sequential processing: Only handles the highest priority intent and queues the rest.
    When multiple intents are detected, we process the highest priority one and save
    the others to the intent queue for subsequent requests.
    """
    s = cast(BrainState, state)
    user_intents = s["user_intents"]
    user_id = s["user_id"]

    if not user_intents:
        raise ValueError("no intents found. something went very wrong.")

    context_pool = list(cast(List[Any] | None, s.get("intents_with_context")) or [])
    continuation_actions = cast(List[str], s.get("continuation_actions") or [])
    intent_context_map = cast(dict[str, str], s.get("intent_context_map") or {})
    updates: Dict[str, Any] = {}

    active_context = _consume_queued_intent(s, user_id, updates, pop_next_intent)
    selection = _select_intent(user_intents, user_id)

    if selection.deferred and "user_intents" not in updates:
        updates["user_intents"] = [selection.primary]

    active_context = _resolve_active_context(
        active_context,
        context_pool=context_pool,
        intent_context_map=intent_context_map,
        primary_intent=selection.primary,
        user_id=user_id,
    )

    if selection.deferred:
        queue_ctx = DeferredQueueContext(
            selection=selection,
            context_pool=context_pool,
            intent_context_map=intent_context_map,
            continuation_actions=continuation_actions,
            intents_with_context=cast(Optional[List[Any]], s.get("intents_with_context")),
            user_id=user_id,
            original_query=s["user_query"],
        )
        _queue_deferred_intents(queue_ctx)

    if not active_context.text:
        fallback_text = s["transformed_query"]
        active_context = ActiveContext(text=fallback_text, source="transformed_query")
        logger.info(
            "[process-intents] No specialized context; using transformed query for user=%s: '%s'",
            _log_safe_user(user_id),
            fallback_text,
        )

    updates["active_intent_context"] = active_context.text
    updates["active_intent_context_source"] = active_context.source
    logger.info(
        "[process-intents] Intent routing context established (source=%s, intent=%s, user=%s)",
        active_context.source,
        selection.primary.value,
        _log_safe_user(user_id),
    )

    node_name = INTENT_NODE_MAP.get(selection.primary)
    if node_name is None:
        raise ValueError(f"Unknown intent type: {selection.primary}")

    next_state = cast(BrainState, {**s, **updates}) if updates else s
    sends = [Send(node_name, next_state)]
    logger.info(
        "[process-intents] Routing to node: %s for user=%s",
        node_name,
        _log_safe_user(user_id),
    )
    return cast(List[Hashable], sends)


def create_brain():
    """Assemble and compile the LangGraph for the BT Servant brain."""

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
        wrap_node_with_timing(
            brain_nodes.start,
            "start_node",
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
        "clear_response_language_node",
        wrap_node_with_timing(brain_nodes.clear_response_language, "clear_response_language_node"),
    )
    builder.add_node(
        "set_agentic_strength_node",
        wrap_node_with_timing(brain_nodes.set_agentic_strength, "set_agentic_strength_node"),
    )
    builder.add_node(
        "query_vector_db_node",
        wrap_node_with_timing(
            brain_nodes.query_vector_db,
            "query_vector_db_node",
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
    builder.add_edge("clear_response_language_node", "translate_responses_node")
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
