"""Tests for the logging helpers with correlation ids."""

import logging
from typing import Any, cast

from bt_servant_engine.core.logging import (
    CorrelationIdFilter,
    bind_correlation_id,
    correlation_id_context,
    get_correlation_id,
    get_logger,
    reset_correlation_id,
)


def test_correlation_filter_attaches_context():
    """Filter should attach the current correlation id onto log records."""
    token = bind_correlation_id("abc123")
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
    finally:
        reset_correlation_id(token)


def test_correlation_context_manager_restores_state():
    """Nested contexts restore the original correlation id."""
    with correlation_id_context("ctx"), correlation_id_context("nested"):
        assert get_correlation_id() == "nested"
    assert get_correlation_id() is None


def test_get_logger_has_correlation_filter():
    """Handlers registered on the logger include the correlation filter."""
    logger = get_logger("bt_servant_engine.tests.logging")
    assert logger.handlers, "logger should have handlers configured"
    assert all(  # each handler should include the correlation filter
        any(isinstance(flt, CorrelationIdFilter) for flt in handler.filters)
        for handler in logger.handlers
    )
