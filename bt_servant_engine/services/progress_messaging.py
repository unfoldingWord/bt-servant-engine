"""Progress messaging service for providing user feedback during long operations.

This module provides infrastructure for sending strategic progress messages to users
during computationally expensive operations, maintaining clean architecture by using
callbacks rather than direct adapter dependencies.
"""

from __future__ import annotations

from time import time
from typing import Any, Awaitable, Callable, cast

from bt_servant_engine.core.logging import get_logger

logger = get_logger(__name__)

# Type alias for progress messenger callback
ProgressMessenger = Callable[[str], Awaitable[None]]


async def maybe_send_progress(
    state: Any,
    message: str,
    force: bool = False,
    min_interval: float = 3.0,
) -> None:
    """Send a progress message if conditions are met.

    Args:
        state: The BrainState dictionary containing progress configuration
        message: The progress message to send to the user
        force: If True, bypass throttling and always send the message
        min_interval: Minimum seconds between messages (default: 3.0)

    Notes:
        - Messages are throttled to avoid spam (default 3 seconds between messages)
        - The force parameter allows critical messages to bypass throttling
        - Failures to send progress messages are logged but don't interrupt processing
        - Progress messaging can be disabled per-request via progress_enabled flag
    """
    # Import here to avoid circular dependency
    from bt_servant_engine.services.brain_orchestrator import BrainState

    s = cast(BrainState, state)

    # Check if progress messaging is enabled for this request
    if not s.get("progress_enabled"):
        return

    # Get the messenger callback
    messenger = s.get("progress_messenger")
    if not messenger:
        logger.debug("No progress messenger configured, skipping message: %s", message)
        return

    # Apply throttling unless forced
    current_time = time()
    last_time = s.get("last_progress_time", 0)
    throttle = s.get("progress_throttle_seconds", min_interval)
    time_since_last = current_time - last_time

    if not force and time_since_last < throttle:
        logger.debug(
            "Throttled progress message: '%s' (only %.1fs since last, need %.1fs)",
            message,
            time_since_last,
            throttle,
        )
        return

    # Send the progress message
    try:
        await messenger(message)
        s["last_progress_time"] = current_time
        logger.info(
            "Sent progress message: '%s' (%.1fs since last%s)",
            message,
            time_since_last,
            " [FORCED]" if force else "",
        )
    except Exception:  # pylint: disable=broad-exception-caught
        # Log but don't interrupt processing if progress message fails
        logger.warning("Failed to send progress message: %s", message, exc_info=True)


def should_show_translation_progress(state: Any) -> bool:
    """Determine if translation progress should be shown.

    Shows progress for:
    - Non-English target languages
    - Long responses (>500 characters)
    - Complex scripts (Arabic, Chinese, Japanese, Hebrew, Hindi)

    Args:
        state: The BrainState dictionary

    Returns:
        True if translation progress should be shown
    """
    from bt_servant_engine.services.brain_orchestrator import BrainState

    s = cast(BrainState, state)
    responses = s.get("responses", [])
    target_lang = s.get("user_response_language")

    # Skip if English (no translation needed)
    if target_lang == "English" or not target_lang:
        return False

    # Check total response length
    total_length = sum(len(r.get("response", "")) for r in responses)

    # Complex scripts need more processing time
    complex_scripts = ["Arabic", "Chinese", "Japanese", "Hebrew", "Hindi"]
    is_complex = target_lang in complex_scripts

    # Show progress for long responses or complex scripts
    return total_length > 500 or is_complex


__all__ = [
    "maybe_send_progress",
    "should_show_translation_progress",
    "ProgressMessenger",
]
