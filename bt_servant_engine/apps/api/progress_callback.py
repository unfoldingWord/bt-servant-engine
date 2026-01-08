"""HTTP callback messenger for progress updates.

This module provides infrastructure for sending progress messages to a
webhook URL during long-running operations. The messenger is async and
fire-and-forget - failures are logged but don't interrupt processing.
"""

from __future__ import annotations

import time
from http import HTTPStatus
from typing import TYPE_CHECKING, Awaitable, Callable

import httpx

from bt_servant_engine.core.logging import get_logger

if TYPE_CHECKING:
    from bt_servant_engine.services.status_messages import LocalizedProgressMessage

logger = get_logger(__name__)

# Fire-and-forget timeout (don't block processing)
CALLBACK_TIMEOUT = 2.0

# HTTP status threshold for error responses
_HTTP_ERROR_THRESHOLD = HTTPStatus.BAD_REQUEST

# Type alias for the progress messenger callable
ProgressMessenger = Callable[["LocalizedProgressMessage"], Awaitable[None]]


async def create_webhook_messenger(
    callback_url: str,
    user_id: str,
    auth_token: str | None = None,
) -> ProgressMessenger:
    """Create a progress messenger that POSTs to a webhook URL.

    The messenger is async and fire-and-forget - failures are logged
    but don't interrupt processing.

    Args:
        callback_url: URL to POST progress messages to.
        user_id: User ID to include in progress payloads.
        auth_token: Optional token to include in X-Engine-Token header.

    Returns:
        An async callable that sends progress messages.
    """

    async def send_progress(message: "LocalizedProgressMessage") -> None:
        payload = {
            "user_id": user_id,
            "message_key": message.get("key", ""),
            "text": message.get("text", ""),
            "timestamp": time.time(),
        }

        headers = {}
        if auth_token:
            headers["X-Engine-Token"] = auth_token

        try:
            async with httpx.AsyncClient(timeout=CALLBACK_TIMEOUT) as client:
                response = await client.post(callback_url, json=payload, headers=headers)
                if response.status_code >= _HTTP_ERROR_THRESHOLD:
                    logger.warning(
                        "Progress callback failed: %s - %s",
                        response.status_code,
                        response.text[:200] if response.text else "(no body)",
                    )
                else:
                    logger.info(
                        "Progress callback sent successfully: key=%s url=%s status=%s user=%s",
                        message.get("key"),
                        callback_url,
                        response.status_code,
                        user_id[:8] + "..." if len(user_id) > 8 else user_id,
                    )
        except httpx.TimeoutException:
            logger.debug("Progress callback timed out (non-blocking): %s", callback_url)
        except httpx.RequestError as exc:
            logger.warning("Progress callback request failed: %s", exc)

    return send_progress


__all__ = ["create_webhook_messenger", "ProgressMessenger", "CALLBACK_TIMEOUT"]
