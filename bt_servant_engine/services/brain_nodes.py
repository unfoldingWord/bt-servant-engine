"""LangGraph node functions for BT Servant brain.

This module contains all node implementations and helper functions used in the
brain decision graph. These are thin wrappers that delegate to service modules
while handling state extraction and dependency injection.
"""
# pylint: disable=too-many-lines

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable, Optional, cast

from openai import OpenAI

from bt_servant_engine.core.config import config
from bt_servant_engine.core.intents import IntentType
from bt_servant_engine.core.logging import get_logger
from bt_servant_engine.core.ports import ChromaPort, UserStatePort
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
from bt_servant_engine.services.brain_followups import FollowupConfig, apply_followups
from bt_servant_engine.services.truncation_notices import build_truncation_notice
from bt_servant_engine.services.continuation_prompts import INTENT_ACTION_DESCRIPTIONS
from bt_servant_engine.services.preprocessing import (
    determine_intents as _determine_intents_impl,
    determine_intents_structured as _determine_intents_structured_impl,
    determine_query_language as _determine_query_language_impl,
    is_affirmative_response_to_continuation as _is_affirmative_response_to_continuation_impl,
    model_for_agentic_strength as _model_for_agentic_strength,
    preprocess_user_query as _preprocess_user_query_impl,
    resolve_agentic_strength as _resolve_agentic_strength,
    resolve_dev_agentic_mcp as _resolve_dev_agentic_mcp,
    generate_continuation_actions as _generate_continuation_actions_impl,
)
from bt_servant_engine.services.passage_selection import (
    PassageSelectionDependencies,
    PassageSelectionRequest,
    resolve_selection_for_single_book as resolve_selection_for_single_book_impl,
)
from bt_servant_engine.services.intents.simple_intents import (
    BOILER_PLATE_AVAILABLE_FEATURES_MESSAGE,
    converse_with_bt_servant as converse_with_bt_servant_impl,
    handle_system_information_request as handle_system_information_request_impl,
    handle_unsupported_function as handle_unsupported_function_impl,
)
from bt_servant_engine.services.intents.settings_intents import (
    AgenticStrengthDependencies,
    AgenticStrengthRequest,
    ClearResponseLanguageDependencies,
    ClearResponseLanguageRequest,
    DevAgenticMCPDependencies,
    DevAgenticMCPRequest,
    ResponseLanguageDependencies,
    ResponseLanguageRequest,
    clear_response_language as clear_response_language_impl,
    set_agentic_strength as set_agentic_strength_impl,
    set_dev_agentic_mcp as set_dev_agentic_mcp_impl,
    set_response_language as set_response_language_impl,
    clear_dev_agentic_mcp as clear_dev_agentic_mcp_impl,
)
from bt_servant_engine.services.intents.passage_intents import (
    ListenToScriptureRequest,
    PassageKeywordsRequest,
    PassageSummaryRequest,
    RetrieveScriptureRequest,
    get_passage_keywords as get_passage_keywords_impl,
    get_passage_summary as get_passage_summary_impl,
    listen_to_scripture as listen_to_scripture_impl,
    retrieve_scripture as retrieve_scripture_impl,
)
from bt_servant_engine.services.intents.translation_intents import (
    TranslationDependencies,
    TranslationHelpsDependencies,
    TranslationHelpsRequestParams,
    TranslationRequestParams,
    get_translation_helps as get_translation_helps_impl,
    translate_scripture as translate_scripture_impl,
)
from bt_servant_engine.services.intents.fia_intents import (
    FIADependencies,
    FIARequest,
    consult_fia_resources as consult_fia_resources_impl,
    FIA_REFERENCE_CONTENT,
)
from bt_servant_engine.services.intent_queue import (
    clear_queue,
    has_queued_intents,
    peek_next_intent,
)
from bt_servant_engine.services.response_pipeline import (
    ChunkingDependencies,
    ChunkingRequest,
    ResponseLocalizationRequest,
    ResponseTranslationDependencies,
    ResponseTranslationRequest,
    chunk_message as chunk_message_impl,
    needs_chunking as needs_chunking_impl,
    reconstruct_structured_text as reconstruct_structured_text_impl,
    translate_or_localize_response as translate_or_localize_response_impl,
    translate_text as translate_text_impl,
    build_translation_queue as build_translation_queue_impl,
    resolve_target_language as resolve_target_language_impl,
)
from bt_servant_engine.services.graph_pipeline import (
    OpenAIQueryDependencies,
    OpenAIQueryPayload,
    query_open_ai as query_open_ai_impl,
    query_vector_db as query_vector_db_impl,
)
from bt_servant_engine.services.translation_helpers import (
    build_translation_helps_context as build_translation_helps_context_impl,
    build_translation_helps_messages as build_translation_helps_messages_impl,
    prepare_translation_helps as prepare_translation_helps_impl,
)
from bt_servant_engine.services.mcp_agentic import (
    MCPAgenticDependencies,
    run_agentic_mcp,
)
from bt_servant_engine.services.status_messages import get_effective_response_language
from bt_servant_engine.services import runtime
from utils.bsb import BOOK_MAP as BSB_BOOK_MAP
from utils.perf import add_tokens
from utils.identifiers import get_log_safe_user_id

# Initialize module-level dependencies
BASE_DIR = Path(__file__).resolve().parent.parent.parent
open_ai_client = OpenAI(api_key=config.OPENAI_API_KEY)
logger = get_logger(__name__)

if TYPE_CHECKING:
    from bt_servant_engine.services.brain_orchestrator import BrainState


def _brain_state(state: Any) -> "BrainState":
    """Return the langgraph state as a typed BrainState for downstream helpers."""
    return cast("BrainState", state)


# Constants
FIRST_INTERACTION_MESSAGE = f"""
Hello! I am the BT Servant. This is our first conversation. Let's work together to understand and translate God's word!

{BOILER_PLATE_AVAILABLE_FEATURES_MESSAGE}
"""

LANG_DETECTION_SAMPLE_CHARS = 100


def _user_state_port() -> UserStatePort:
    services = runtime.get_services()
    if services.user_state is None:
        raise RuntimeError("User state port is not configured.")
    return services.user_state


def _chroma_port() -> ChromaPort:
    services = runtime.get_services()
    if services.chroma is None:
        raise RuntimeError("Chroma port is not configured.")
    return services.chroma


# Response helper wrappers


def _intent_query_for_node(state: Any, node_name: str) -> str:
    """Return the intent-specific query text for the active node with logging."""
    s = _brain_state(state)
    context = cast(Optional[str], s.get("active_intent_context"))
    source = cast(str, s.get("active_intent_context_source", "unknown"))
    transformed = s.get("transformed_query", "")
    if not context:
        intent_context_map = cast(dict[str, str], s.get("intent_context_map", {}))
        active_intents = cast(Optional[list], s.get("user_intents"))
        if active_intents:
            active_intent = active_intents[0]
            intent_value = (
                active_intent.value if hasattr(active_intent, "value") else str(active_intent)
            )
            mapped = intent_context_map.get(intent_value)
            if mapped:
                context = mapped
                source = "intent_context_map"

    if context:
        logger.info(
            "[intent-context] Node=%s will use active context (source=%s): '%s'",
            node_name,
            source,
            context,
        )
        return context

    logger.warning(
        "[intent-context] Node=%s missing active context; falling back to transformed query: '%s'",
        node_name,
        transformed,
    )
    return transformed


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
    state: Any,
    protected_items: list[dict],
    normal_items: list[dict],
) -> list[dict | str]:
    """Assemble responses in the order they should be translated or localized."""
    return build_translation_queue_impl(state, protected_items, normal_items)


def _resolve_target_language(
    state: Any,
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
    request = ResponseLocalizationRequest(
        client=open_ai_client,
        response=resp,
        target_language=target_language,
        agentic_strength=agentic_strength,
    )
    dependencies = ResponseTranslationDependencies(
        model_for_agentic_strength=_model_for_agentic_strength,
        extract_cached_input_tokens=_extract_cached_input_tokens,
    )
    return translate_or_localize_response_impl(request, dependencies)


# Graph node functions


def start(state: Any) -> dict:
    """Handle first interaction greeting, otherwise no-op."""
    s = _brain_state(state)
    user_id = s["user_id"]
    user_state = _user_state_port()
    if user_state.is_first_interaction(user_id):
        user_state.set_first_interaction(user_id, False)
        return {
            "responses": [{"intent": "first-interaction", "response": FIRST_INTERACTION_MESSAGE}]
        }
    return {}


def determine_intents(state: Any) -> dict:
    """Classify the user's transformed query into one or more intents.

    Flow:
    1. Check for queued intents first (user continuing from previous request)
    2. If queue exists, use LLM to check if user is responding affirmatively
    3. If affirmative, use queued intent; if not, clear queue and detect new intent
    4. If no queue, detect intents using simple classification
    5. If multiple intents detected, use structured extraction for parameter disambiguation
    6. If single intent, skip extraction (whole query is context - no need for LLM overhead)
    """
    s = _brain_state(state)
    query = s["transformed_query"]
    queued_result = _resolve_queued_intent_response(s, query)
    if queued_result:
        return queued_result

    logger.info("[determine-intents] Detecting intents from query: %s", query[:100])
    user_intents = _determine_intents_impl(open_ai_client, query)

    if len(user_intents) > 1:
        return _handle_multiple_detected_intents(s, query, user_intents)

    # Single intent - no extraction needed, whole query is context
    logger.info(
        "[determine-intents] Single intent detected=%s; skipping parameter extraction",
        user_intents[0].value if user_intents else "none",
    )
    context_map_single = {user_intents[0].value: query} if user_intents else {}
    return {
        "user_intents": user_intents,
        "intent_context_map": context_map_single,
        # Handlers derive context directly from the transformed query
    }


def _resolve_queued_intent_response(state: "BrainState", query: str) -> dict | None:
    """Return queued intent metadata when the user affirms a continuation prompt."""
    user_id = cast(Optional[str], state.get("user_id"))
    if not user_id or not has_queued_intents(user_id):
        return None

    next_item = peek_next_intent(user_id)
    if not next_item:
        return None

    action_description = INTENT_ACTION_DESCRIPTIONS.get(
        next_item.intent, f"help with {next_item.intent.value.replace('-', ' ')}"
    )
    logger.info(
        "[determine-intents] Queued intent=%s (%s); checking for affirmative response",
        next_item.intent.value,
        action_description,
    )

    is_affirmative = _is_affirmative_response_to_continuation_impl(
        open_ai_client, query, action_description
    )
    if is_affirmative:
        logger.info(
            "[determine-intents] Affirmative reply detected; resuming queued intent=%s",
            next_item.intent.value,
        )
        return {
            "user_intents": [next_item.intent],
            "queued_intent_context": next_item.context_text,
            "intent_context_map": {next_item.intent.value: next_item.context_text},
        }

    log_user_id = get_log_safe_user_id(user_id, secret=config.LOG_PSEUDONYM_SECRET)
    logger.info(
        "[determine-intents] User declined continuation; clearing queue for user=%s",
        log_user_id,
    )
    clear_queue(user_id)
    return None


def _handle_multiple_detected_intents(
    state: "BrainState",
    query: str,
    detected_intents: list,
) -> dict:
    """Return structured context for multi-intent queries and follow-up prompts."""
    logger.info(
        "[determine-intents] Multiple intents detected (%d); running structured extraction",
        len(detected_intents),
    )
    intents_with_context = _determine_intents_structured_impl(
        open_ai_client, query, list(detected_intents)
    )
    intent_types = [intent_context.intent for intent_context in intents_with_context]
    context_map = {
        intent_context.intent.value: intent_context.trimmed_context()
        for intent_context in intents_with_context
    }
    logger.info(
        "[determine-intents] Structured extraction complete; storing context for %d intents",
        len(intents_with_context),
    )
    for index, intent_context in enumerate(intents_with_context, start=1):
        logger.info(
            "[determine-intents]   Intent %d context='%s' (intent=%s)",
            index,
            intent_context.trimmed_context(),
            intent_context.intent.value,
        )

    logger.info("[determine-intents] Building continuation prompts for multi-intent query")
    target_language = get_effective_response_language(state)
    continuation_actions = _generate_continuation_actions_impl(
        open_ai_client, query, intent_types, target_language
    )
    return {
        "user_intents": intent_types,
        "intents_with_context": intents_with_context,
        "intent_context_map": context_map,
        "continuation_actions": continuation_actions,
    }


def set_response_language(state: Any) -> dict:
    """Detect and persist the user's desired response language."""

    s = _brain_state(state)
    intent_query = _intent_query_for_node(state, "set_response_language_node")
    user_state = _user_state_port()
    request = ResponseLanguageRequest(
        client=open_ai_client,
        user_id=s["user_id"],
        user_query=intent_query,
        chat_history=s["user_chat_history"],
    )
    dependencies = ResponseLanguageDependencies(
        set_user_response_language=user_state.set_response_language,
    )
    return set_response_language_impl(request, dependencies)


def clear_response_language(state: Any) -> dict:
    """Clear the user's stored response language preference."""

    s = _brain_state(state)
    user_state = _user_state_port()
    request = ClearResponseLanguageRequest(user_id=s["user_id"])
    dependencies = ClearResponseLanguageDependencies(
        clear_user_response_language=user_state.clear_response_language
    )
    return clear_response_language_impl(request, dependencies)


def set_agentic_strength(state: Any) -> dict:
    """Detect and persist the user's preferred agentic strength."""

    s = _brain_state(state)
    intent_query = _intent_query_for_node(state, "set_agentic_strength_node")
    user_state = _user_state_port()
    request = AgenticStrengthRequest(
        client=open_ai_client,
        user_id=s["user_id"],
        user_query=intent_query,
        chat_history=s["user_chat_history"],
        log_pseudonym_secret=config.LOG_PSEUDONYM_SECRET,
    )
    dependencies = AgenticStrengthDependencies(
        set_user_agentic_strength=user_state.set_agentic_strength
    )
    return set_agentic_strength_impl(request, dependencies)


def set_dev_agentic_mcp(state: Any) -> dict:
    """Enable the developer MCP agentic mode for the user."""

    s = _brain_state(state)
    user_state = _user_state_port()
    request = DevAgenticMCPRequest(user_id=s["user_id"])
    dependencies = DevAgenticMCPDependencies(
        set_user_dev_agentic_mcp=user_state.set_dev_agentic_mcp
    )
    return set_dev_agentic_mcp_impl(request, dependencies)


def clear_dev_agentic_mcp(state: Any) -> dict:
    """Disable the developer MCP agentic mode for the user."""

    s = _brain_state(state)
    user_state = _user_state_port()
    request = DevAgenticMCPRequest(user_id=s["user_id"])
    dependencies = DevAgenticMCPDependencies(
        set_user_dev_agentic_mcp=user_state.set_dev_agentic_mcp
    )
    return clear_dev_agentic_mcp_impl(request, dependencies)


def _collect_truncation_notices(
    protected_items: Iterable[dict[str, Any]],
    normal_items: Iterable[dict[str, Any]],
) -> list[Optional[dict[str, Any]]]:
    notices: list[Optional[dict[str, Any]]] = []

    def _extract(item: dict[str, Any]) -> Optional[dict[str, Any]]:
        notice = item.get("truncation_notice")
        if not isinstance(notice, dict):
            return None
        delivered_raw = notice.get("delivered_label")
        delivered_label = str(delivered_raw).strip() if delivered_raw is not None else ""
        if not delivered_label:
            return None
        verse_limit = int(notice.get("verse_limit", config.TRANSLATION_HELPS_VERSE_LIMIT))
        original_raw = notice.get("original_label")
        original_label = (
            str(original_raw).strip()
            if isinstance(original_raw, str) and original_raw.strip()
            else None
        )
        return {
            "verse_limit": verse_limit,
            "delivered_label": delivered_label,
            "original_label": original_label,
        }

    for item in protected_items:
        notices.append(_extract(item))
    for item in normal_items:
        notices.append(_extract(item))
    return notices


def _translate_queue_with_notices(
    queue: list[dict | str],
    protected_items: Iterable[dict[str, Any]],
    normal_items: Iterable[dict[str, Any]],
    target_language: str,
    agentic_strength: str,
) -> tuple[list[str], Callable[[str, str], str]]:
    translate_fn = partial(translate_text, agentic_strength=agentic_strength)
    raw_translated = [
        _translate_or_localize_response(resp, target_language, agentic_strength) for resp in queue
    ]

    translated: list[str] = []
    for item in raw_translated:
        translated.append(item if isinstance(item, str) else str(item))

    notice_specs = _collect_truncation_notices(protected_items, normal_items)
    for idx, notice in enumerate(notice_specs):
        if not notice:
            continue
        notice_text = build_truncation_notice(
            language=target_language,
            verse_limit=notice["verse_limit"],
            delivered_label=notice["delivered_label"],
            original_label=notice["original_label"],
            translate_text_fn=translate_fn,
        )
        if notice_text:
            translated[idx] = f"{translated[idx]}\n\n{notice_text}"

    return translated, translate_fn


def translate_responses(state: Any) -> dict:
    """Translate or localize responses into the user's desired language."""

    s = _brain_state(state)
    raw_responses = [
        resp for resp in cast(list[dict], s["responses"]) if not resp.get("suppress_text_delivery")
    ]
    if not raw_responses:
        if bool(s.get("send_voice_message")):
            logger.info("[translate] skipping text translation because delivery is voice-only")
            return {"translated_responses": [], "final_response_language": None}
        raise ValueError("no responses to translate. something bad happened. bailing out.")

    protected_items, normal_items = _partition_response_items(raw_responses)
    queue = _build_translation_queue(s, protected_items, normal_items)
    if not queue:
        if bool(s.get("send_voice_message")):
            logger.info("[translate] no text responses after queue assembly; voice-only delivery")
            return {"translated_responses": [], "final_response_language": None}
        raise ValueError("no responses to translate. something bad happened. bailing out.")

    target_language, passthrough = _resolve_target_language(s, queue)
    if passthrough is not None:
        return {"translated_responses": passthrough, "final_response_language": target_language}
    if target_language is None:
        raise RuntimeError("target language resolution failed")

    agentic_strength = _resolve_agentic_strength(cast(dict[str, Any], s))
    translated_responses, translate_fn = _translate_queue_with_notices(
        queue,
        protected_items,
        normal_items,
        target_language,
        agentic_strength,
    )

    updates = apply_followups(
        s,
        translated_responses,
        raw_responses,
        FollowupConfig(
            target_language=target_language,
            agentic_strength=agentic_strength,
            translate_text=translate_fn,
            followup_already_added=bool(s.get("followup_question_added", False)),
        ),
    )

    if "passage_followup_context" in s:
        updates.setdefault("passage_followup_context", {})

    base_response = {
        "translated_responses": translated_responses,
        "final_response_language": target_language,
    }
    if updates:
        base_response.update(updates)
        return base_response
    return base_response


def translate_text(
    response_text: str,
    target_language: str,
    *,
    agentic_strength: Optional[str] = None,
) -> str:
    """Translate a single text into the target ISO 639-1 language code."""
    request = ResponseTranslationRequest(
        client=open_ai_client,
        text=response_text,
        target_language=target_language,
        agentic_strength=agentic_strength,
    )
    dependencies = ResponseTranslationDependencies(
        model_for_agentic_strength=_model_for_agentic_strength,
        extract_cached_input_tokens=_extract_cached_input_tokens,
    )
    return translate_text_impl(request, dependencies)


def determine_query_language(state: Any) -> dict:
    """Determine the language of the user's original query and set collection order."""

    s = _brain_state(state)
    query = s["user_query"]
    agentic_strength = _resolve_agentic_strength(cast(dict[str, Any], s))
    query_language, stack_rank_collections = _determine_query_language_impl(
        open_ai_client, query, agentic_strength
    )
    return {"query_language": query_language, "stack_rank_collections": stack_rank_collections}


def preprocess_user_query(state: Any) -> dict:
    """Lightly clarify or correct the user's query using conversation history."""

    s = _brain_state(state)
    query = s["user_query"]
    chat_history = s["user_chat_history"]
    transformed_query, _, _ = _preprocess_user_query_impl(open_ai_client, query, chat_history)
    return {"transformed_query": transformed_query}


def query_vector_db(state: Any) -> dict:
    """Query the vector DB (Chroma) across ranked collections and filter by relevance."""

    s = _brain_state(state)
    intent_query = _intent_query_for_node(state, "query_vector_db_node")
    return query_vector_db_impl(
        intent_query,
        s["stack_rank_collections"],
        _chroma_port().get_collection,
        BOILER_PLATE_AVAILABLE_FEATURES_MESSAGE,
    )


def query_open_ai(state: Any) -> dict:
    """Generate the final response text using RAG context and OpenAI."""

    s = _brain_state(state)
    intent_query = _intent_query_for_node(state, "query_open_ai_node")
    agentic_strength = _resolve_agentic_strength(cast(dict[str, Any], s))
    payload = OpenAIQueryPayload(
        docs=s["docs"],
        transformed_query=intent_query,
        chat_history=s["user_chat_history"],
        agentic_strength=agentic_strength,
        boilerplate_features_message=BOILER_PLATE_AVAILABLE_FEATURES_MESSAGE,
    )
    dependencies = OpenAIQueryDependencies(
        model_for_agentic_strength=_model_for_agentic_strength,
        extract_cached_input_tokens=_extract_cached_input_tokens,
        add_tokens=add_tokens,
    )
    return query_open_ai_impl(open_ai_client, payload, dependencies)


def consult_fia_resources(state: Any) -> dict:
    """Answer FIA-specific questions using FIA collections and reference material."""

    s = _brain_state(state)
    intent_query = _intent_query_for_node(state, "consult_fia_resources_node")
    agentic_strength = _resolve_agentic_strength(cast(dict[str, Any], s))
    request = FIARequest(
        client=open_ai_client,
        query=intent_query,
        chat_history=s["user_chat_history"],
        user_response_language=s.get("user_response_language"),
        query_language=s.get("query_language"),
        agentic_strength=agentic_strength,
    )
    dependencies = FIADependencies(
        get_chroma_collection=_chroma_port().get_collection,
        model_for_agentic_strength=_model_for_agentic_strength,
        extract_cached_input_tokens=_extract_cached_input_tokens,
    )
    return consult_fia_resources_impl(request, dependencies)


def chunk_message(state: Any) -> dict:
    """Chunk oversized responses to respect WhatsApp limits, via LLM or fallback."""

    s = _brain_state(state)
    responses = s["translated_responses"]
    text_to_chunk = responses[0]
    chunk_max = config.MAX_RESPONSE_CHUNK_SIZE - 100
    request = ChunkingRequest(
        client=open_ai_client,
        text_to_chunk=text_to_chunk,
        additional_responses=responses[1:],
        chunk_max=chunk_max,
    )
    dependencies = ChunkingDependencies(
        extract_cached_input_tokens=_extract_cached_input_tokens,
    )
    chunks = chunk_message_impl(request, dependencies)
    return {
        "translated_responses": chunks,
        "final_response_language": s.get("final_response_language"),
    }


def needs_chunking(state: Any) -> str:
    """Return next node key if chunking is required, otherwise finish."""
    return needs_chunking_impl(state)


def handle_unsupported_function(state: Any) -> dict:
    """Generate a helpful response when the user requests unsupported functionality."""

    s = _brain_state(state)
    intent_query = _intent_query_for_node(state, "handle_unsupported_function_node")
    return handle_unsupported_function_impl(open_ai_client, intent_query, s["user_chat_history"])


def handle_system_information_request(state: Any) -> dict:
    """Provide help/about information for the BT Servant system."""

    s = _brain_state(state)
    intent_query = _intent_query_for_node(state, "handle_system_information_request_node")
    return handle_system_information_request_impl(
        open_ai_client, intent_query, s["user_chat_history"]
    )


def converse_with_bt_servant(state: Any) -> dict:
    """Respond conversationally to the user based on context and history."""

    s = _brain_state(state)
    intent_query = _intent_query_for_node(state, "converse_with_bt_servant_node")
    return converse_with_bt_servant_impl(open_ai_client, intent_query, s["user_chat_history"])


# Passage helper wrappers


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


def _build_selection_dependencies() -> PassageSelectionDependencies:
    """Construct shared dependencies for passage selection helpers."""
    return PassageSelectionDependencies(
        client=open_ai_client,
        book_map=BSB_BOOK_MAP,
        detect_mentioned_books=_detect_mentioned_books,
        translate_text=translate_text,
    )


def resolve_selection_for_single_book(
    query: str,
    query_lang: str,
    focus_hint: str | None = None,
) -> tuple[str | None, list[tuple[int, int | None, int | None, int | None]] | None, str | None]:
    """Parse and normalize a user query into a single canonical book and ranges."""
    dependencies = _build_selection_dependencies()
    request = PassageSelectionRequest(
        query=query,
        query_lang=query_lang,
        dependencies=dependencies,
        focus_hint=focus_hint,
    )
    return resolve_selection_for_single_book_impl(request)


# Passage intent node functions


def handle_get_passage_summary(state: Any) -> dict:
    """Handle get-passage-summary: extract refs, retrieve verses, summarize."""

    s = _brain_state(state)
    intent_query = _intent_query_for_node(state, "handle_get_passage_summary_node")
    dev_agentic_mcp = _resolve_dev_agentic_mcp(cast(dict[str, Any], s))
    if dev_agentic_mcp:
        agentic_deps = MCPAgenticDependencies(
            openai_client=open_ai_client,
            extract_cached_tokens_fn=_extract_cached_input_tokens,
            brain_state=cast(dict[str, Any], s),
        )
        logger.info("[agentic-mcp] using dev MCP flow for get-passage-summary")
        response_text = run_agentic_mcp(
            agentic_deps,
            user_message=intent_query,
            intent=IntentType.GET_PASSAGE_SUMMARY,
        )
        return {
            "responses": [{"intent": IntentType.GET_PASSAGE_SUMMARY, "response": response_text}]
        }

    agentic_strength = _resolve_agentic_strength(cast(dict[str, Any], s))
    selection_request = PassageSelectionRequest(
        query=intent_query,
        query_lang=s["query_language"],
        dependencies=_build_selection_dependencies(),
    )
    request = PassageSummaryRequest(
        selection=selection_request,
        user_response_language=s.get("user_response_language"),
        agentic_strength=agentic_strength,
        model_for_agentic_strength=_model_for_agentic_strength,
        extract_cached_input_tokens=_extract_cached_input_tokens,
    )
    return get_passage_summary_impl(request)


def handle_get_passage_keywords(state: Any) -> dict:
    """Handle get-passage-keywords: extract refs, retrieve keywords, and list them."""

    s = _brain_state(state)
    intent_query = _intent_query_for_node(state, "handle_get_passage_keywords_node")
    dev_agentic_mcp = _resolve_dev_agentic_mcp(cast(dict[str, Any], s))
    if dev_agentic_mcp:
        agentic_deps = MCPAgenticDependencies(
            openai_client=open_ai_client,
            extract_cached_tokens_fn=_extract_cached_input_tokens,
            brain_state=cast(dict[str, Any], s),
        )
        logger.info("[agentic-mcp] using dev MCP flow for get-passage-keywords")
        response_text = run_agentic_mcp(
            agentic_deps,
            user_message=intent_query,
            intent=IntentType.GET_PASSAGE_KEYWORDS,
        )
        return {
            "responses": [{"intent": IntentType.GET_PASSAGE_KEYWORDS, "response": response_text}]
        }

    selection_request = PassageSelectionRequest(
        query=intent_query,
        query_lang=s["query_language"],
        dependencies=_build_selection_dependencies(),
    )
    request = PassageKeywordsRequest(selection=selection_request)
    return get_passage_keywords_impl(request)


def handle_get_translation_helps(state: Any) -> dict:
    """Generate focused translation helps guidance for a selected passage."""

    s = _brain_state(state)
    intent_query = _intent_query_for_node(state, "handle_get_translation_helps_node")
    agentic_strength = _resolve_agentic_strength(cast(dict[str, Any], s))
    dev_agentic_mcp = _resolve_dev_agentic_mcp(cast(dict[str, Any], s))

    if dev_agentic_mcp:
        agentic_deps = MCPAgenticDependencies(
            openai_client=open_ai_client,
            extract_cached_tokens_fn=_extract_cached_input_tokens,
            brain_state=cast(dict[str, Any], s),
        )
        logger.info("[agentic-mcp] using dev MCP flow for get-translation-helps")
        response_text = run_agentic_mcp(
            agentic_deps,
            user_message=intent_query,
            intent=IntentType.GET_TRANSLATION_HELPS,
        )
        return {
            "responses": [{"intent": IntentType.GET_TRANSLATION_HELPS, "response": response_text}]
        }

    request = TranslationHelpsRequestParams(
        client=open_ai_client,
        query=intent_query,
        query_lang=s["query_language"],
        book_map=BSB_BOOK_MAP,
        agentic_strength=agentic_strength,
    )
    dependencies = TranslationHelpsDependencies(
        detect_books_fn=_detect_mentioned_books,
        translate_text_fn=translate_text,
        select_model_fn=_model_for_agentic_strength,
        extract_cached_tokens_fn=_extract_cached_input_tokens,
        prepare_translation_helps_fn=prepare_translation_helps_impl,
        build_context_fn=build_translation_helps_context_impl,
        build_messages_fn=build_translation_helps_messages_impl,
    )
    return get_translation_helps_impl(request, dependencies)


def handle_retrieve_scripture(state: Any) -> dict:
    """Handle retrieve-scripture with optional auto-translation."""

    s = _brain_state(state)
    intent_query = _intent_query_for_node(state, "handle_retrieve_scripture_node")
    dev_agentic_mcp = _resolve_dev_agentic_mcp(cast(dict[str, Any], s))
    if dev_agentic_mcp:
        agentic_deps = MCPAgenticDependencies(
            openai_client=open_ai_client,
            extract_cached_tokens_fn=_extract_cached_input_tokens,
            brain_state=cast(dict[str, Any], s),
        )
        logger.info("[agentic-mcp] using dev MCP flow for retrieve-scripture")
        response_text = run_agentic_mcp(
            agentic_deps,
            user_message=intent_query,
            intent=IntentType.RETRIEVE_SCRIPTURE,
        )
        return {"responses": [{"intent": IntentType.RETRIEVE_SCRIPTURE, "response": response_text}]}

    agentic_strength = _resolve_agentic_strength(cast(dict[str, Any], s))
    selection_request = PassageSelectionRequest(
        query=intent_query,
        query_lang=s["query_language"],
        dependencies=_build_selection_dependencies(),
    )
    request = RetrieveScriptureRequest(
        selection=selection_request,
        user_response_language=s.get("user_response_language"),
        agentic_strength=agentic_strength,
        extract_cached_input_tokens=_extract_cached_input_tokens,
    )
    return retrieve_scripture_impl(request)


def handle_listen_to_scripture(state: Any) -> dict:
    """Delegate to retrieve-scripture and request voice delivery."""

    s = _brain_state(state)
    intent_query = _intent_query_for_node(state, "handle_listen_to_scripture_node")
    agentic_strength = _resolve_agentic_strength(cast(dict[str, Any], s))
    selection_request = PassageSelectionRequest(
        query=intent_query,
        query_lang=s["query_language"],
        dependencies=_build_selection_dependencies(),
    )
    retrieve_request = RetrieveScriptureRequest(
        selection=selection_request,
        user_response_language=s.get("user_response_language"),
        agentic_strength=agentic_strength,
        extract_cached_input_tokens=_extract_cached_input_tokens,
    )
    request = ListenToScriptureRequest(
        retrieve_request=retrieve_request,
        reconstruct_structured_text=_reconstruct_structured_text,
    )
    return listen_to_scripture_impl(request)


def handle_translate_scripture(state: Any) -> dict:
    """Handle translate-scripture: return verses translated into a target language."""

    s = _brain_state(state)
    intent_query = _intent_query_for_node(state, "handle_translate_scripture_node")
    agentic_strength = _resolve_agentic_strength(cast(dict[str, Any], s))
    request = TranslationRequestParams(
        client=open_ai_client,
        query=intent_query,
        query_lang=s["query_language"],
        book_map=BSB_BOOK_MAP,
        user_response_language=s.get("user_response_language"),
        agentic_strength=agentic_strength,
    )
    dependencies = TranslationDependencies(
        detect_books_fn=_detect_mentioned_books,
        translate_text_fn=translate_text,
        select_model_fn=_model_for_agentic_strength,
        extract_cached_tokens_fn=_extract_cached_input_tokens,
    )
    return translate_scripture_impl(request, dependencies)


# Language detection helpers


def _sample_for_language_detection(text: str) -> str:
    """Return a short prefix ending at a whitespace boundary for detection."""
    return _sample_for_language_detection_impl(text, LANG_DETECTION_SAMPLE_CHARS)


__all__ = [
    # Dependencies (for test compatibility)
    "open_ai_client",
    "FIA_REFERENCE_CONTENT",
    # Node functions
    "start",
    "determine_intents",
    "set_response_language",
    "set_agentic_strength",
    "set_dev_agentic_mcp",
    "clear_dev_agentic_mcp",
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
    # Helper functions (for test compatibility)
    "resolve_selection_for_single_book",
    "clear_response_language",
]
