"""Generate context-aware continuation prompts for queued intents.

This module creates continuation prompts that guide users through their queued intents,
encouraging them to continue the conversation in a natural way.
"""

from __future__ import annotations

from typing import Any, Optional, cast

from bt_servant_engine.core.config import config
from bt_servant_engine.core.intents import IntentType
from bt_servant_engine.core.logging import get_logger
from bt_servant_engine.services.intent_queue import peek_next_intent
from utils.identifiers import get_log_safe_user_id

logger = get_logger(__name__)

# Map intents to user-friendly action descriptions
INTENT_ACTION_DESCRIPTIONS = {
    IntentType.GET_BIBLE_TRANSLATION_ASSISTANCE: "provide Bible translation assistance",
    IntentType.CONSULT_FIA_RESOURCES: "consult FIA resources",
    IntentType.GET_PASSAGE_SUMMARY: "summarize that passage",
    IntentType.GET_PASSAGE_KEYWORDS: "show key terms from that passage",
    IntentType.GET_TRANSLATION_HELPS: "provide translation helps for that passage",
    IntentType.RETRIEVE_SCRIPTURE: "retrieve that scripture passage",
    IntentType.LISTEN_TO_SCRIPTURE: "read that passage aloud",
    IntentType.TRANSLATE_SCRIPTURE: "translate that scripture",
    IntentType.PERFORM_UNSUPPORTED_FUNCTION: "help with that request",
    IntentType.RETRIEVE_SYSTEM_INFORMATION: "provide system information",
    IntentType.SET_RESPONSE_LANGUAGE: "set your response language",
    IntentType.CLEAR_RESPONSE_LANGUAGE: "clear your response language preference",
    IntentType.SET_AGENTIC_STRENGTH: "adjust your agentic strength preference",
    IntentType.CONVERSE_WITH_BT_SERVANT: "continue our conversation",
}


def generate_continuation_prompt(user_id: str, state: Any) -> Optional[str]:
    """Generate a continuation prompt if the user has queued intents.

    Args:
        user_id: The user's identifier
        state: BrainState dictionary holding language metadata for logging

    Returns:
        A continuation prompt string (complete question), or None if no queued intents
    """
    response_language = None
    if isinstance(state, dict):
        response_language = cast(Optional[str], state.get("user_response_language"))
    log_user_id = get_log_safe_user_id(user_id, secret=config.LOG_PSEUDONYM_SECRET)
    logger.debug(
        "[continuation-prompt] Checking queued intents for user=%s (resp_language=%s)",
        log_user_id,
        response_language,
    )

    # Check if there's a next intent in the queue
    next_item = peek_next_intent(user_id)
    if not next_item:
        logger.debug("[continuation-prompt] No queued intents for user=%s", log_user_id)
        return None

    # Use pre-generated continuation question if available (should be a complete question now)
    if not next_item.continuation_action:
        logger.warning(
            "[continuation-prompt] Missing continuation question for user=%s "
            "(intent=%s, context='%s'); skipping prompt",
            log_user_id,
            next_item.intent.value,
            next_item.context_text,
        )
        return None

    logger.info(
        "[continuation-prompt] Using queued continuation question for user=%s: '%s' (intent=%s)",
        log_user_id,
        next_item.continuation_action,
        next_item.intent.value,
    )

    # Prepend newlines for spacing
    prompt = f"\n\n{next_item.continuation_action}"

    logger.info(
        "[continuation-prompt] Generated prompt for user=%s: intent=%s",
        log_user_id,
        next_item.intent.value,
    )

    return prompt


def generate_continuation_prompt_with_context(
    user_id: str, next_intent: IntentType, parameters: Optional[dict] = None
) -> str:
    """Generate a continuation prompt with specific context from parameters.

    This is a more sophisticated version that incorporates extracted parameters
    into the continuation prompt. For example, if the next intent is GET_PASSAGE_SUMMARY
    with parameters {"passage": "Romans 8"}, we can say:
    "Would you like me to summarize Romans 8?"

    Args:
        user_id: The user's identifier
        next_intent: The next intent to process
        parameters: Extracted parameters for the intent (optional)

    Returns:
        A context-aware continuation prompt string
    """
    log_user_id = get_log_safe_user_id(user_id, secret=config.LOG_PSEUDONYM_SECRET)
    logger.debug(
        "[continuation-prompt-context] Generating for user=%s, intent=%s, params=%s",
        log_user_id,
        next_intent.value,
        parameters,
    )

    # Get base action description
    base_action = INTENT_ACTION_DESCRIPTIONS.get(
        next_intent, f"help with {next_intent.value.replace('-', ' ')}"
    )

    # Try to enhance with parameters
    if parameters:
        passage = parameters.get("passage")
        if passage and next_intent in {
            IntentType.GET_PASSAGE_SUMMARY,
            IntentType.GET_PASSAGE_KEYWORDS,
            IntentType.GET_TRANSLATION_HELPS,
            IntentType.RETRIEVE_SCRIPTURE,
            IntentType.LISTEN_TO_SCRIPTURE,
            IntentType.TRANSLATE_SCRIPTURE,
        }:
            # Replace "that passage" with specific passage reference
            base_action = base_action.replace("that passage", passage)

        target_language = parameters.get("target_language")
        if target_language and next_intent == IntentType.TRANSLATE_SCRIPTURE:
            base_action = f"translate {passage or 'that scripture'} to {target_language}"

    prompt = f"\n\nWould you like me to {base_action}?"

    logger.info(
        "[continuation-prompt-context] Generated context-aware prompt for user=%s: '%s'",
        log_user_id,
        base_action,
    )

    return prompt


__all__ = [
    "generate_continuation_prompt",
    "generate_continuation_prompt_with_context",
    "INTENT_ACTION_DESCRIPTIONS",
]
