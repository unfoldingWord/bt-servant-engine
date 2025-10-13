"""Intent queue management for sequential multi-intent processing.

This module handles persistent storage and retrieval of intent queues in the user_state
JSON DB. When multiple intents are detected, we process the highest priority one first
and queue the rest with their pre-extracted parameters for later processing.
"""

from __future__ import annotations

import time
from typing import Optional

from tinydb import Query
from tinydb.operations import delete

from bt_servant_engine.adapters.user_state import get_user_db, get_user_state, set_user_state
from bt_servant_engine.core.intents import IntentQueue, IntentQueueItem
from bt_servant_engine.core.logging import get_logger

logger = get_logger(__name__)

# Queue expiry time: 10 minutes
QUEUE_TTL_SECONDS = 10 * 60

# Maximum number of queued intents to prevent unbounded growth
MAX_QUEUE_SIZE = 5


def load_intent_queue(user_id: str) -> Optional[IntentQueue]:
    """Load the intent queue for a user from persistent storage.

    Args:
        user_id: The user's identifier

    Returns:
        IntentQueue if one exists and hasn't expired, None otherwise
    """
    logger.info("[intent-queue] Loading queue for user=%s", user_id)

    try:
        user_state = get_user_state(user_id)
        queue_data = user_state.get("intent_queue")

        if not queue_data:
            logger.info("[intent-queue] No queue found for user=%s", user_id)
            return None

        queue = IntentQueue(**queue_data)

        # Check expiry
        now = time.time()
        if now > queue.expires_at:
            logger.info(
                "[intent-queue] Queue expired for user=%s (expired_at=%f, now=%f, age=%d seconds)",
                user_id,
                queue.expires_at,
                now,
                int(now - queue.expires_at + QUEUE_TTL_SECONDS),
            )
            clear_queue(user_id)
            return None

        logger.info(
            "[intent-queue] Loaded queue for user=%s with %d items (expires in %d seconds)",
            user_id,
            len(queue.items),
            int(queue.expires_at - now),
        )

        # Log each queued item for visibility
        for idx, item in enumerate(queue.items):
            logger.info(
                "[intent-queue]   Item %d: intent=%s, params=%s, age=%d seconds",
                idx,
                item.intent.value,
                item.parameters,
                int(now - item.created_at),
            )

        return queue

    except Exception:  # pylint: disable=broad-except
        logger.error("[intent-queue] Error loading queue for user=%s", user_id, exc_info=True)
        return None


def save_intent_queue(
    user_id: str,
    intents: list[IntentQueueItem],
) -> None:
    """Save an intent queue to persistent storage.

    Args:
        user_id: The user's identifier
        intents: List of intent items to queue
    """
    # Enforce max queue size
    if len(intents) > MAX_QUEUE_SIZE:
        logger.warning(
            "[intent-queue] Queue size (%d) exceeds max (%d) for user=%s, truncating",
            len(intents),
            MAX_QUEUE_SIZE,
            user_id,
        )
        intents = intents[:MAX_QUEUE_SIZE]

    logger.info("[intent-queue] Saving queue for user=%s with %d items", user_id, len(intents))

    # Log what we're queueing
    for idx, item in enumerate(intents):
        logger.info(
            "[intent-queue]   Queueing item %d: intent=%s, params=%s, query=%s",
            idx,
            item.intent.value,
            item.parameters,
            item.original_query[:50] + "..."
            if len(item.original_query) > 50
            else item.original_query,
        )

    try:
        queue = IntentQueue(
            items=intents,
            expires_at=time.time() + QUEUE_TTL_SECONDS,
        )

        user_state = get_user_state(user_id)
        user_state["intent_queue"] = queue.model_dump()
        set_user_state(user_id, user_state)

        logger.info(
            "[intent-queue] Successfully saved queue for user=%s (expires in %d seconds)",
            user_id,
            QUEUE_TTL_SECONDS,
        )

    except Exception:  # pylint: disable=broad-except
        logger.error("[intent-queue] Error saving queue for user=%s", user_id, exc_info=True)


def pop_next_intent(user_id: str) -> Optional[IntentQueueItem]:
    """Pop the next intent from the queue.

    Args:
        user_id: The user's identifier

    Returns:
        The next IntentQueueItem if queue exists and has items, None otherwise
    """
    logger.info("[intent-queue] Popping next intent for user=%s", user_id)

    queue = load_intent_queue(user_id)
    if not queue or not queue.items:
        logger.info("[intent-queue] No intents to pop for user=%s", user_id)
        return None

    # Pop the first item
    next_item = queue.items[0]
    remaining_items = queue.items[1:]

    logger.info(
        "[intent-queue] Popped intent=%s with params=%s for user=%s",
        next_item.intent.value,
        next_item.parameters,
        user_id,
    )

    # Save remaining items or clear if empty
    if remaining_items:
        logger.info(
            "[intent-queue] %d items remain in queue for user=%s", len(remaining_items), user_id
        )
        save_intent_queue(user_id, remaining_items)
    else:
        logger.info("[intent-queue] Queue now empty for user=%s, clearing", user_id)
        clear_queue(user_id)

    return next_item


def peek_next_intent(user_id: str) -> Optional[IntentQueueItem]:
    """Peek at the next intent without removing it from the queue.

    Args:
        user_id: The user's identifier

    Returns:
        The next IntentQueueItem if queue exists and has items, None otherwise
    """
    logger.debug("[intent-queue] Peeking at next intent for user=%s", user_id)

    queue = load_intent_queue(user_id)
    if not queue or not queue.items:
        return None

    next_item = queue.items[0]
    logger.debug(
        "[intent-queue] Next intent is %s with params=%s for user=%s",
        next_item.intent.value,
        next_item.parameters,
        user_id,
    )

    return next_item


def has_queued_intents(user_id: str) -> bool:
    """Check if user has any queued intents.

    Args:
        user_id: The user's identifier

    Returns:
        True if queue exists and has items, False otherwise
    """
    queue = load_intent_queue(user_id)
    has_items = queue is not None and len(queue.items) > 0

    logger.debug(
        "[intent-queue] User %s %s queued intents",
        user_id,
        "has" if has_items else "does not have",
    )

    return has_items


def get_queue_size(user_id: str) -> int:
    """Get the number of queued intents for a user.

    Args:
        user_id: The user's identifier

    Returns:
        Number of queued intents (0 if no queue)
    """
    queue = load_intent_queue(user_id)
    size = len(queue.items) if queue else 0

    logger.debug("[intent-queue] Queue size for user=%s is %d", user_id, size)

    return size


def clear_queue(user_id: str) -> None:
    """Clear the intent queue for a user.

    Args:
        user_id: The user's identifier
    """
    logger.info("[intent-queue] Clearing queue for user=%s", user_id)

    try:
        user_state = get_user_state(user_id)
        if "intent_queue" in user_state:
            # Use TinyDB's delete() operation to actually remove the field
            # upsert() uses merge semantics and won't delete fields not in the dict
            q = Query()
            get_user_db().table("users").update(delete("intent_queue"), q.user_id == user_id)

            logger.info("[intent-queue] Successfully cleared queue for user=%s", user_id)
        else:
            logger.debug("[intent-queue] No queue to clear for user=%s", user_id)

    except Exception:  # pylint: disable=broad-except
        logger.error("[intent-queue] Error clearing queue for user=%s", user_id, exc_info=True)


__all__ = [
    "load_intent_queue",
    "save_intent_queue",
    "pop_next_intent",
    "peek_next_intent",
    "has_queued_intents",
    "get_queue_size",
    "clear_queue",
]
