"""Per-user message locking to serialize concurrent requests.

This module provides asyncio-based per-user locks to ensure that messages
from the same user are processed sequentially, preventing race conditions
when users send multiple messages in quick succession.

The lock manager includes TTL-based cleanup to prevent memory leaks from
accumulating locks for inactive users.
"""

from __future__ import annotations

import asyncio
from time import time

from bt_servant_engine.core.logging import get_logger

logger = get_logger(__name__)

# Lock storage with last-access tracking
_user_locks: dict[str, asyncio.Lock] = {}
_user_lock_last_used: dict[str, float] = {}
_meta_lock = asyncio.Lock()  # Protects the dictionaries

LOCK_TTL_SECONDS = 600  # 10 minutes


async def get_user_lock(user_id: str) -> asyncio.Lock:
    """Get or create a lock for a user.

    Args:
        user_id: The user identifier to get a lock for.

    Returns:
        An asyncio.Lock for the specified user.
    """
    async with _meta_lock:
        if user_id not in _user_locks:
            _user_locks[user_id] = asyncio.Lock()
        _user_lock_last_used[user_id] = time()
        return _user_locks[user_id]


async def cleanup_stale_locks() -> int:
    """Remove locks for users inactive for longer than TTL.

    Only removes locks that are not currently held.

    Returns:
        The number of locks removed.
    """
    async with _meta_lock:
        now = time()
        stale = [
            uid
            for uid, last in _user_lock_last_used.items()
            if now - last > LOCK_TTL_SECONDS and not _user_locks[uid].locked()
        ]
        for uid in stale:
            del _user_locks[uid]
            del _user_lock_last_used[uid]
        return len(stale)


async def lock_cleanup_task() -> None:
    """Background task to periodically clean up stale locks.

    This task runs indefinitely, cleaning up locks every 5 minutes.
    It should be started on app startup and cancelled on shutdown.
    """
    while True:
        await asyncio.sleep(300)  # Every 5 minutes
        try:
            removed = await cleanup_stale_locks()
            if removed:
                logger.info("Cleaned up %d stale user locks", removed)
        except Exception:  # pylint: disable=broad-exception-caught
            # Intentionally broad: background task must not crash
            logger.exception("Error during lock cleanup")


def get_lock_stats() -> dict[str, int]:
    """Get current lock statistics for monitoring.

    Returns:
        Dictionary with lock count and held lock count.
    """
    return {
        "total_locks": len(_user_locks),
        "held_locks": sum(1 for lock in _user_locks.values() if lock.locked()),
    }


def clear_all_locks() -> None:
    """Clear all locks. Only for testing purposes."""
    _user_locks.clear()
    _user_lock_last_used.clear()


__all__ = [
    "get_user_lock",
    "cleanup_stale_locks",
    "lock_cleanup_task",
    "get_lock_stats",
    "clear_all_locks",
    "LOCK_TTL_SECONDS",
]
