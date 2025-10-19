"""Tests for the logging helpers with correlation ids."""

import logging
from typing import Any, cast

from pythonjsonlogger import jsonlogger

from bt_servant_engine.core.logging import (
    CorrelationIdFilter,
    bind_client_ip,
    bind_correlation_id,
    bind_log_user_id,
    client_ip_context,
    correlation_id_context,
    get_client_ip,
    get_correlation_id,
    get_log_user_id,
    get_logger,
    log_user_id_context,
    reset_client_ip,
    reset_correlation_id,
    reset_log_user_id,
)


def test_correlation_filter_attaches_context():
    """Filter should attach the current correlation id onto log records."""
    cid_token = bind_correlation_id("abc123")
    user_token = bind_log_user_id("safe-user")
    ip_token = bind_client_ip("203.0.113.10")
    try:
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname=__file__,
            lineno=10,
            msg="hello",
            args=None,
            exc_info=None,
        )
        filt = CorrelationIdFilter()
        assert filt.filter(record) is True
        record_any = cast(Any, record)
        assert "correlation_id" in record_any.__dict__
        assert record_any.__dict__["correlation_id"] == "abc123"
        assert record_any.__dict__["log_user_id"] == "safe-user"
        assert record_any.__dict__["client_ip"] == "203.0.113.10"
    finally:
        reset_client_ip(ip_token)
        reset_log_user_id(user_token)
        reset_correlation_id(cid_token)


def test_correlation_context_manager_restores_state():
    """Nested contexts restore the original correlation id."""
    with (
        correlation_id_context("ctx"),
        correlation_id_context("nested"),
        log_user_id_context("user-a"),
        client_ip_context("198.51.100.8"),
    ):
        assert get_correlation_id() == "nested"
        assert get_log_user_id() == "user-a"
        assert get_client_ip() == "198.51.100.8"
    assert get_correlation_id() is None
    assert get_log_user_id() is None
    assert get_client_ip() is None


def test_get_logger_has_correlation_filter():
    """Handlers registered on the logger include the correlation filter."""
    logger = get_logger("bt_servant_engine.tests.logging")
    assert logger.handlers, "logger should have handlers configured"
    assert all(  # each handler should include the correlation filter
        any(isinstance(flt, CorrelationIdFilter) for flt in handler.filters)
        for handler in logger.handlers
    )
    assert all(
        isinstance(handler.formatter, jsonlogger.JsonFormatter) for handler in logger.handlers
    )
