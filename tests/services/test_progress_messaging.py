"""Tests for progress messaging service."""

import asyncio

from bt_servant_engine.services import status_messages
from bt_servant_engine.services.progress_messaging import (
    maybe_send_progress,
    should_show_translation_progress,
)


def test_should_show_translation_progress_english():
    """Test that translation progress is not shown for English."""
    state = {
        "responses": [{"response": "A long response" * 100}],
        "user_response_language": "English",
    }

    assert not should_show_translation_progress(state)


def test_should_show_translation_progress_no_language():
    """Test that translation progress is not shown when no language set."""
    state = {
        "responses": [{"response": "A long response" * 100}],
        "user_response_language": None,
    }

    assert not should_show_translation_progress(state)


def test_should_show_translation_progress_complex_script():
    """Test that translation progress is shown for complex scripts."""
    state = {
        "responses": [{"response": "Short response"}],
        "user_response_language": "Arabic",
    }

    assert should_show_translation_progress(state)


def test_should_show_translation_progress_long_response():
    """Test that translation progress is shown for long responses."""
    state = {
        "responses": [{"response": "A" * 600}],
        "user_response_language": "Spanish",
    }

    assert should_show_translation_progress(state)


def test_should_show_translation_progress_short_simple():
    """Test that translation progress is not shown for short simple responses."""
    state = {
        "responses": [{"response": "Short"}],
        "user_response_language": "Spanish",
    }

    assert not should_show_translation_progress(state)


# Tests for maybe_send_progress function


def test_maybe_send_progress_disabled():
    """Test that progress messages are not sent when disabled."""

    async def run_test():
        state = {
            "progress_enabled": False,
            "progress_messenger": None,
        }

        # Should return early without calling messenger
        await maybe_send_progress(state, status_messages.make_progress_message("Test message"))
        # No assertion needed - just verify it doesn't crash

    asyncio.run(run_test())


def test_maybe_send_progress_no_messenger():
    """Test that progress messages are skipped when no messenger configured."""

    async def run_test():
        state = {
            "progress_enabled": True,
            "progress_messenger": None,
        }

        # Should return early without calling messenger
        await maybe_send_progress(state, status_messages.make_progress_message("Test message"))
        # No assertion needed - just verify it doesn't crash

    asyncio.run(run_test())


def test_maybe_send_progress_throttled():
    """Test that progress messages are throttled."""

    async def run_test():
        calls = []

        async def mock_messenger(msg):
            calls.append(msg)

        state = {
            "progress_enabled": True,
            "progress_messenger": mock_messenger,
            "last_progress_time": 9999999999.0,  # Very recent time
            "progress_throttle_seconds": 3.0,
        }

        # Should be throttled
        await maybe_send_progress(
            state, status_messages.make_progress_message("Test message"), force=False
        )
        assert len(calls) == 0  # Message not sent

    asyncio.run(run_test())


def test_maybe_send_progress_forced():
    """Test that forced messages bypass throttling."""

    async def run_test():
        calls = []

        async def mock_messenger(msg):
            calls.append(msg)

        state = {
            "progress_enabled": True,
            "progress_messenger": mock_messenger,
            "last_progress_time": 9999999999.0,  # Very recent time
            "progress_throttle_seconds": 3.0,
        }

        # Should bypass throttling with force=True
        await maybe_send_progress(
            state, status_messages.make_progress_message("Forced message"), force=True
        )
        assert len(calls) == 1
        assert calls[0]["text"] == "_Forced message_"

    asyncio.run(run_test())


def test_maybe_send_progress_success():
    """Test successful progress message sending."""

    async def run_test():
        calls = []

        async def mock_messenger(msg):
            calls.append(msg)

        state = {
            "progress_enabled": True,
            "progress_messenger": mock_messenger,
            "last_progress_time": 0,  # Long time ago
            "progress_throttle_seconds": 3.0,
        }

        await maybe_send_progress(state, status_messages.make_progress_message("Success message"))
        assert len(calls) == 1
        assert calls[0]["text"] == "_Success message_"
        assert state["last_progress_time"] > 0  # Time was updated

    asyncio.run(run_test())


def test_maybe_send_progress_messenger_failure():
    """Test that messenger failures don't crash the system."""

    async def run_test():
        async def failing_messenger(msg):
            raise Exception("Messenger failed")  # pylint: disable=broad-exception-raised

        state = {
            "progress_enabled": True,
            "progress_messenger": failing_messenger,
            "last_progress_time": 0,
            "progress_throttle_seconds": 3.0,
        }

        # Should not raise exception
        await maybe_send_progress(state, status_messages.make_progress_message("Test message"))
        # Just verify it doesn't crash

    asyncio.run(run_test())
