"""Generate context-aware continuation prompts for queued intents.

This module creates continuation prompts that guide users through their queued intents,
encouraging them to continue the conversation in a natural way.
"""

from __future__ import annotations

from typing import Optional

from bt_servant_engine.core.intents import IntentType
from bt_servant_engine.core.logging import get_logger
from bt_servant_engine.services.intent_queue import peek_next_intent

logger = get_logger(__name__)

# Map intents to user-friendly action descriptions
INTENT_ACTION_DESCRIPTIONS = {
    IntentType.GET_BIBLE_TRANSLATION_ASSISTANCE: "provide Bible translation assistance",
    IntentType.CONSULT_FIA_RESOURCES: "consult FIA resources",
    IntentType.GET_PASSAGE_SUMMARY: "summarize that passage",
    IntentType.GET_PASSAGE_KEYWORDS: "show key terms from that passage",
    IntentType.GET_TRANSLATION_HELPS: "provide translation helps",
    IntentType.RETRIEVE_SCRIPTURE: "retrieve that scripture passage",
    IntentType.LISTEN_TO_SCRIPTURE: "read that passage aloud",
    IntentType.TRANSLATE_SCRIPTURE: "translate that scripture",
    IntentType.PERFORM_UNSUPPORTED_FUNCTION: "help with that request",
    IntentType.RETRIEVE_SYSTEM_INFORMATION: "provide system information",
    IntentType.SET_RESPONSE_LANGUAGE: "set your response language",
    IntentType.SET_AGENTIC_STRENGTH: "adjust your agentic strength preference",
    IntentType.CONVERSE_WITH_BT_SERVANT: "continue our conversation",
}


def generate_continuation_prompt(user_id: str) -> Optional[str]:
    """Generate a continuation prompt if the user has queued intents.

    Args:
        user_id: The user's identifier

    Returns:
        A continuation prompt string, or None if no queued intents
    """
    logger.debug("[continuation-prompt] Checking for queued intents for user=%s", user_id)

    # Check if there's a next intent in the queue
    next_item = peek_next_intent(user_id)
    if not next_item:
        logger.debug("[continuation-prompt] No queued intents for user=%s", user_id)
        return None

    # Get the action description for this intent
    base_action = INTENT_ACTION_DESCRIPTIONS.get(
        next_item.intent, f"help with {next_item.intent.value.replace('-', ' ')}"
    )

    # Try to enhance with parameters for specificity
    action = base_action
    params = next_item.parameters

    if params:
        passage = params.get("passage")
        if passage and next_item.intent in {
            IntentType.GET_PASSAGE_SUMMARY,
            IntentType.GET_PASSAGE_KEYWORDS,
            IntentType.GET_TRANSLATION_HELPS,
            IntentType.RETRIEVE_SCRIPTURE,
            IntentType.LISTEN_TO_SCRIPTURE,
            IntentType.TRANSLATE_SCRIPTURE,
        }:
            # Replace "that passage" with specific passage reference
            action = base_action.replace("that passage", passage)

        target_language = params.get("target_language")
        if target_language and next_item.intent == IntentType.TRANSLATE_SCRIPTURE:
            action = f"translate {passage or 'that scripture'} to {target_language}"

    # Build the continuation prompt
    prompt = f"\n\nWould you like me to {action}?"

    logger.info(
        "[continuation-prompt] Generated prompt for user=%s: intent=%s, action='%s'",
        user_id,
        next_item.intent.value,
        action,
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
    logger.debug(
        "[continuation-prompt-context] Generating for user=%s, intent=%s, params=%s",
        user_id,
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
        user_id,
        base_action,
    )

    return prompt


__all__ = [
    "generate_continuation_prompt",
    "generate_continuation_prompt_with_context",
    "INTENT_ACTION_DESCRIPTIONS",
]
