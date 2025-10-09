"""Tests for progress messaging service."""

from bt_servant_engine.services.progress_messaging import should_show_translation_progress


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
