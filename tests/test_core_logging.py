"""Tests for the logging helpers with correlation ids."""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, cast

from pythonjsonlogger import jsonlogger

from bt_servant_engine.core.logging import (
    CorrelationIdFilter,
    LOG_FILE_PATH,
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
    root_logger = logging.getLogger()
    handlers = [
        handler
        for handler in root_logger.handlers
        if (
            isinstance(handler, RotatingFileHandler)
            and Path(getattr(handler, "baseFilename", "")) == LOG_FILE_PATH
        )
        or (
            isinstance(handler, logging.StreamHandler)
            and getattr(handler, "stream", None) is sys.stdout
        )
    ]
    assert handlers, "expected shared stream/file handlers to be installed"
    assert not logger.handlers, "module logger should rely on shared handlers"
    assert all(  # each handler should include the correlation filter
        any(isinstance(flt, CorrelationIdFilter) for flt in handler.filters) for handler in handlers
    )
    assert all(isinstance(handler.formatter, jsonlogger.JsonFormatter) for handler in handlers)


def test_get_logger_uses_shared_rotating_handler():
    """Calling get_logger repeatedly should not duplicate file handlers."""
    first = get_logger("bt_servant_engine.tests.logging.first")
    second = get_logger("bt_servant_engine.tests.logging.second")

    assert not first.handlers
    assert not second.handlers

    root_handlers = logging.getLogger().handlers
    rotating_handlers = [
        handler for handler in root_handlers if isinstance(handler, RotatingFileHandler)
    ]
    assert len(rotating_handlers) == 1, "expected exactly one shared RotatingFileHandler"
