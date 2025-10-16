"""Tests for log-safe identifier helper functions."""

from utils.identifiers import clear_log_safe_user_cache, get_log_safe_user_id

PSEUDONYM_LENGTH = 16


def test_get_log_safe_user_id_is_deterministic(monkeypatch) -> None:
    """Same user id + secret should always yield the same pseudonym."""
    monkeypatch.setenv("LOG_PSEUDONYM_SECRET", "deterministic-secret")
    clear_log_safe_user_cache()

    token_first = get_log_safe_user_id("+15551234567")
    token_second = get_log_safe_user_id("+15551234567")

    assert token_first == token_second
    assert len(token_first) == PSEUDONYM_LENGTH
    clear_log_safe_user_cache()


def test_get_log_safe_user_id_varies_by_input_and_secret(monkeypatch) -> None:
    """Changing the user id or secret should change the pseudonym output."""
    monkeypatch.setenv("LOG_PSEUDONYM_SECRET", "deterministic-secret")
    clear_log_safe_user_cache()
    base_token = get_log_safe_user_id("+15551234567")

    clear_log_safe_user_cache()
    monkeypatch.setenv("LOG_PSEUDONYM_SECRET", "alternate-secret")
    different_secret_token = get_log_safe_user_id("+15551234567")

    clear_log_safe_user_cache()
    monkeypatch.setenv("LOG_PSEUDONYM_SECRET", "deterministic-secret")
    different_user_token = get_log_safe_user_id("+15557654321")

    assert base_token != different_secret_token
    assert base_token != different_user_token
    clear_log_safe_user_cache()
