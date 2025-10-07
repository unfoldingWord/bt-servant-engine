"""Tests for the logging helpers with correlation ids and request context."""

import json
import logging
from typing import Any, cast

from bt_servant_engine.core.logging import (
    ContextFilter,
    bind_correlation_id,
    bind_request_context,
    correlation_id_context,
    get_request_context,
    get_correlation_id,
    get_logger,
    request_context,
    reset_correlation_id,
    reset_request_context,
    JsonLogFormatter,
)
from bt_servant_engine.core.models import RequestContext


def test_context_filter_attaches_correlation_and_request_context():
    """Filter should attach the current correlation id and request context."""
    token = bind_correlation_id("abc123")
    ctx = RequestContext(correlation_id="abc123", path="/foo", method="GET", user_agent=None)
    ctx_token = bind_request_context(ctx)
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
        filt = ContextFilter()
        assert filt.filter(record) is True
        record_any = cast(Any, record)
        assert "correlation_id" in record_any.__dict__
        assert record_any.__dict__["correlation_id"] == "abc123"
        assert record_any.__dict__["request_context"] is ctx
    finally:
        reset_request_context(ctx_token)
        reset_correlation_id(token)


def test_correlation_context_manager_restores_state():
    """Nested contexts restore the original correlation id."""
    with correlation_id_context("ctx"), correlation_id_context("nested"):
        assert get_correlation_id() == "nested"
    assert get_correlation_id() is None


def test_request_context_manager_restores_state():
    """Request context manager should restore prior context."""
    outer = RequestContext(correlation_id="outer", path="/outer", method="GET")
    inner = RequestContext(correlation_id="inner", path="/inner", method="POST")
    assert get_request_context() is None
    with request_context(outer):
        assert get_request_context() is outer
        with request_context(inner):
            assert get_request_context() is inner
        assert get_request_context() is outer
    assert get_request_context() is None


def test_json_formatter_serializes_expected_fields():
    """Json formatter should include base fields and http context."""
    formatter = JsonLogFormatter()
    record = logging.LogRecord(
        name="bt.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=42,
        msg="hello",
        args=None,
        exc_info=None,
    )
    ctx = RequestContext(correlation_id="cid", path="/alive", method="GET", user_agent="test-agent")
    record.correlation_id = "cid"
    record.request_context = ctx
    payload = json.loads(formatter.format(record))
    assert payload["correlation_id"] == "cid"
    assert payload["message"] == "hello"
    assert payload["http"]["path"] == "/alive"
    assert payload["http"]["method"] == "GET"
    assert payload["http"]["user_agent"] == "test-agent"


def test_get_logger_has_correlation_filter():
    """Handlers registered on the logger include the correlation filter."""
    logger = get_logger("bt_servant_engine.tests.logging")
    assert logger.handlers, "logger should have handlers configured"
    assert all(  # each handler should include the correlation filter
        any(isinstance(flt, ContextFilter) for flt in handler.filters)
        for handler in logger.handlers
    )
