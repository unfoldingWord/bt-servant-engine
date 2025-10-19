"""Intent queue management for sequential multi-intent processing.

Queues are persisted via the configured :class:`UserStatePort` implementation.
This module no longer reaches directly into adapters; instead it resolves the
port from :mod:`bt_servant_engine.services.runtime`.
"""

from __future__ import annotations

import time
from typing import Optional

from bt_servant_engine.core.config import config
from bt_servant_engine.core.intents import IntentQueue, IntentQueueItem
from bt_servant_engine.core.logging import get_logger
from bt_servant_engine.core.ports import UserStatePort
from bt_servant_engine.services import runtime
from utils.identifiers import get_log_safe_user_id

logger = get_logger(__name__)

# Queue expiry time: 10 minutes
QUEUE_TTL_SECONDS = 10 * 60

# Maximum number of queued intents to prevent unbounded growth
MAX_QUEUE_SIZE = 5
# Maximum characters to log/save from the original query when previewing
QUERY_PREVIEW_MAX_LENGTH = 50


def _user_state_port() -> UserStatePort:
    services = runtime.get_services()
    if services.user_state is None:
        raise RuntimeError("User state port has not been configured.")
    return services.user_state


def _log_safe_user(user_id: str) -> str:
    return get_log_safe_user_id(user_id, secret=config.LOG_PSEUDONYM_SECRET)


def load_intent_queue(user_id: str) -> Optional[IntentQueue]:
    """Load the intent queue for a user from persistent storage."""
    log_user_id = _log_safe_user(user_id)
    logger.info("[intent-queue] Loading queue for user=%s", log_user_id)

    try:
        user_state = dict(_user_state_port().load_user_state(user_id))
        queue_data = user_state.get("intent_queue")
        if not queue_data:
            logger.info("[intent-queue] No queue found for user=%s", log_user_id)
            return None

        queue = IntentQueue(**queue_data)

        now = time.time()
        if now > queue.expires_at:
            logger.info(
                "[intent-queue] Queue expired for user=%s (expired_at=%f, now=%f, age=%d seconds)",
                log_user_id,
                queue.expires_at,
                now,
                int(now - queue.expires_at + QUEUE_TTL_SECONDS),
            )
            clear_queue(user_id)
            return None

        logger.info(
            "[intent-queue] Loaded queue for user=%s with %d items (expires in %d seconds)",
            log_user_id,
            len(queue.items),
            int(queue.expires_at - now),
        )
        for idx, item in enumerate(queue.items):
            logger.info(
                "[intent-queue]   Item %d: intent=%s, context='%s', age=%d seconds",
                idx,
                item.intent.value,
                item.context_text,
                int(now - item.created_at),
            )
        return queue
    except Exception:  # pylint: disable=broad-except
        logger.error("[intent-queue] Error loading queue for user=%s", log_user_id, exc_info=True)
        return None


def save_intent_queue(user_id: str, intents: list[IntentQueueItem]) -> None:
    """Save an intent queue to persistent storage."""
    log_user_id = _log_safe_user(user_id)
    if len(intents) > MAX_QUEUE_SIZE:
        logger.warning(
            "[intent-queue] Queue size (%d) exceeds max (%d) for user=%s, truncating",
            len(intents),
            MAX_QUEUE_SIZE,
            log_user_id,
        )
        intents = intents[:MAX_QUEUE_SIZE]

    logger.info("[intent-queue] Saving queue for user=%s with %d items", log_user_id, len(intents))
    for idx, item in enumerate(intents):
        logger.info(
            "[intent-queue]   Queueing item %d: intent=%s, context='%s', query=%s",
            idx,
            item.intent.value,
            item.context_text,
            item.original_query[:QUERY_PREVIEW_MAX_LENGTH] + "..."
            if len(item.original_query) > QUERY_PREVIEW_MAX_LENGTH
            else item.original_query,
        )

    try:
        queue = IntentQueue(items=intents, expires_at=time.time() + QUEUE_TTL_SECONDS)
        port = _user_state_port()
        state = dict(port.load_user_state(user_id))
        state["intent_queue"] = queue.model_dump()
        port.save_user_state(user_id, state)

        logger.info(
            "[intent-queue] Successfully saved queue for user=%s (expires in %d seconds)",
            log_user_id,
            QUEUE_TTL_SECONDS,
        )
    except Exception:  # pylint: disable=broad-except
        logger.error("[intent-queue] Error saving queue for user=%s", log_user_id, exc_info=True)


def pop_next_intent(user_id: str) -> Optional[IntentQueueItem]:
    """Pop the next intent from the queue."""
    log_user_id = _log_safe_user(user_id)
    logger.info("[intent-queue] Popping next intent for user=%s", log_user_id)

    queue = load_intent_queue(user_id)
    if not queue or not queue.items:
        logger.info("[intent-queue] No intents to pop for user=%s", log_user_id)
        return None

    next_item = queue.items[0]
    remaining_items = queue.items[1:]
    logger.info(
        "[intent-queue] Popped intent=%s with context='%s' for user=%s",
        next_item.intent.value,
        next_item.context_text,
        log_user_id,
    )

    if remaining_items:
        logger.info(
            "[intent-queue] %d items remain in queue for user=%s",
            len(remaining_items),
            log_user_id,
        )
        save_intent_queue(user_id, remaining_items)
    else:
        logger.info("[intent-queue] Queue now empty for user=%s, clearing", log_user_id)
        clear_queue(user_id)

    return next_item


def peek_next_intent(user_id: str) -> Optional[IntentQueueItem]:
    """Peek at the next intent without removing it from the queue."""
    log_user_id = _log_safe_user(user_id)
    logger.debug("[intent-queue] Peeking at next intent for user=%s", log_user_id)

    queue = load_intent_queue(user_id)
    if not queue or not queue.items:
        return None

    next_item = queue.items[0]
    logger.debug(
        "[intent-queue] Next intent is %s with context='%s' for user=%s",
        next_item.intent.value,
        next_item.context_text,
        log_user_id,
    )
    return next_item


def has_queued_intents(user_id: str) -> bool:
    """Return whether a queue with items exists for ``user_id``."""
    queue = load_intent_queue(user_id)
    has_items = bool(queue and queue.items)
    log_user_id = _log_safe_user(user_id)
    logger.debug(
        "[intent-queue] User %s %s queued intents",
        log_user_id,
        "has" if has_items else "does not have",
    )
    return has_items


def get_queue_size(user_id: str) -> int:
    """Return the number of queued intents for ``user_id``."""
    queue = load_intent_queue(user_id)
    size = len(queue.items) if queue else 0
    log_user_id = _log_safe_user(user_id)
    logger.debug("[intent-queue] Queue size for user=%s is %d", log_user_id, size)
    return size


def clear_queue(user_id: str) -> None:
    """Clear the intent queue for ``user_id`` if one exists."""
    log_user_id = _log_safe_user(user_id)
    logger.info("[intent-queue] Clearing queue for user=%s", log_user_id)
    try:
        port = _user_state_port()
        state = dict(port.load_user_state(user_id))
        if "intent_queue" in state:
            # Some UserStatePort implementations merge updates rather than replacing the
            # stored dict outright. Set the field to None before saving so that the queue
            # is effectively cleared regardless of merge semantics.
            state["intent_queue"] = None
            port.save_user_state(user_id, state)
            logger.info("[intent-queue] Successfully cleared queue for user=%s", log_user_id)
        else:
            logger.debug("[intent-queue] No queue to clear for user=%s", log_user_id)
    except Exception:  # pylint: disable=broad-except
        logger.error("[intent-queue] Error clearing queue for user=%s", log_user_id, exc_info=True)


__all__ = [
    "load_intent_queue",
    "save_intent_queue",
    "pop_next_intent",
    "peek_next_intent",
    "has_queued_intents",
    "get_queue_size",
    "clear_queue",
]
