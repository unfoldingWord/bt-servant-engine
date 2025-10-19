"""Helper utilities for appending follow-up prompts to translated responses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

from bt_servant_engine.core.config import config as app_config
from bt_servant_engine.core.intents import IntentType
from bt_servant_engine.core.logging import get_logger
from bt_servant_engine.services.continuation_prompts import generate_continuation_prompt
from bt_servant_engine.services.intents.followup_questions import add_followup_if_needed
from bt_servant_engine.services.passage_followups import build_followup_question
from utils.identifiers import get_log_safe_user_id

logger = get_logger(__name__)


@dataclass(slots=True)
class FollowupConfig:
    """Configuration needed to manage follow-up prompts."""

    target_language: str
    agentic_strength: str
    translate_text: Callable[[str, str], str]
    followup_already_added: bool = False


def apply_followups(
    state: Mapping[str, Any],
    translated_responses: list[str],
    raw_responses: Sequence[Mapping[str, Any]],
    config: FollowupConfig,
) -> Dict[str, Any]:
    """Append continuation or intent-specific follow-ups when appropriate."""
    updates: Dict[str, Any] = {}
    followup_added = config.followup_already_added or bool(
        state.get("followup_question_added", False)
    )

    followup_added = _apply_continuation_prompt(
        state,
        translated_responses,
        followup_added,
        updates,
        config,
    )
    if followup_added or not translated_responses or not raw_responses:
        return updates

    intent = raw_responses[-1].get("intent")
    if intent is None:
        return updates

    added = _apply_intent_followup(
        state,
        translated_responses,
        intent,
        config,
    )
    if added:
        updates["followup_question_added"] = True
    return updates


def _apply_continuation_prompt(
    state: Mapping[str, Any],
    translated_responses: list[str],
    followup_added: bool,
    updates: Dict[str, Any],
    config: FollowupConfig,
) -> bool:
    """Append queued-intent continuation prompts to the final response."""
    user_id = state.get("user_id")
    if not user_id:
        return followup_added

    log_user_id = get_log_safe_user_id(user_id, secret=app_config.LOG_PSEUDONYM_SECRET)
    continuation_prompt = generate_continuation_prompt(user_id, state)
    if continuation_prompt and translated_responses:
        prompt_to_append = continuation_prompt
        target_language = (config.target_language or "").strip().lower()
        if target_language and target_language != "en":
            stripped_prompt = continuation_prompt.lstrip()
            prefix_length = len(continuation_prompt) - len(stripped_prompt)
            prefix = continuation_prompt[:prefix_length]
            prompt_body = stripped_prompt
            if prompt_body:
                try:
                    translated_body = config.translate_text(prompt_body, target_language)
                except Exception:  # pylint: disable=broad-except
                    logger.exception(
                        "[translate] Failed to translate continuation prompt for user=%s "
                        "(language=%s); using original text",
                        log_user_id,
                        target_language,
                    )
                else:
                    prompt_to_append = f"{prefix}{translated_body}"
        translated_responses[-1] = translated_responses[-1] + prompt_to_append
        updates["followup_question_added"] = True
        logger.info(
            "[translate] Appended continuation prompt for user=%s",
            log_user_id,
        )
        return True
    return followup_added


def _apply_intent_followup(
    state: Mapping[str, Any],
    translated_responses: list[str],
    intent: IntentType | str,
    config: FollowupConfig,
) -> bool:
    """Append intent-specific follow-up questions when applicable."""
    typed_intent = _normalize_intent(intent)
    if typed_intent is None:
        logger.debug("[translate] Skipping follow-up for unknown intent=%s", intent)
        return False
    if typed_intent in {
        IntentType.CONVERSE_WITH_BT_SERVANT,
        IntentType.RETRIEVE_SYSTEM_INFORMATION,
        IntentType.PERFORM_UNSUPPORTED_FUNCTION,
    }:
        return False

    custom_followup = _custom_followup_for_intent(
        state,
        typed_intent,
        config.target_language,
        config.translate_text,
    )
    translate_fn: Optional[Callable[[str, str], str]] = None
    if custom_followup is None:
        translate_fn = config.translate_text

    updated_response, added_flag = add_followup_if_needed(
        translated_responses[-1],
        state,
        typed_intent,
        custom_followup=custom_followup,
        translate_text_fn=translate_fn,
    )
    translated_responses[-1] = updated_response
    if added_flag:
        logger.info(
            "[translate] Added intent-specific follow-up for intent=%s",
            typed_intent.value,
        )
    return added_flag


def _custom_followup_for_intent(
    state: Mapping[str, Any],
    intent: IntentType,
    target_language: str,
    translate_text_fn: Callable[[str, str], str],
) -> Optional[str]:
    """Return a deterministic follow-up question for passage-based intents."""
    suggestion_ctx = state.get("passage_followup_context")
    if not isinstance(suggestion_ctx, Mapping) or suggestion_ctx.get("intent") != intent:
        return None
    lang_code = str(target_language).strip().lower()
    translate_fn: Optional[Callable[[str, str], str]] = None
    if lang_code != "en":
        translate_fn = translate_text_fn
    return build_followup_question(
        intent,
        suggestion_ctx,
        target_language,
        translate_fn,
    )


def _normalize_intent(intent: IntentType | str) -> Optional[IntentType]:
    """Convert intent representations to IntentType, skipping unknown inputs."""
    if isinstance(intent, IntentType):
        return intent
    try:
        return IntentType(intent)
    except (ValueError, TypeError):
        return None


__all__ = [
    "FollowupConfig",
    "apply_followups",
]
