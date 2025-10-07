"""Structured logging helpers with correlation id and request context support."""

from __future__ import annotations

import json
import logging
import sys
from contextlib import contextmanager
from contextvars import ContextVar, Token
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

from bt_servant_engine.core.config import settings
from bt_servant_engine.core.models import RequestContext

_correlation_id: ContextVar[Optional[str]] = ContextVar("correlation_id", default=None)
_request_context: ContextVar[Optional[RequestContext]] = ContextVar("request_context", default=None)

LEVEL_NAME = str(getattr(settings, "BT_SERVANT_LOG_LEVEL", "info")).upper()
LOG_LEVEL = getattr(logging, LEVEL_NAME, logging.INFO)

BASE_DIR = Path(__file__).resolve().parent.parent
LOGS_DIR = BASE_DIR / "logs"
LOGS_DIR.mkdir(exist_ok=True)
LOG_FILE_PATH = LOGS_DIR / "bt_servant.log"


class ContextFilter(logging.Filter):  # pylint: disable=too-few-public-methods
    """Attach correlation id and request context to log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.correlation_id = get_correlation_id() or "-"
        record.request_context = get_request_context()
        return True


class JsonLogFormatter(logging.Formatter):
    """Serialize log records as structured JSON."""

    def format(self, record: logging.LogRecord) -> str:  # noqa: D401
        base: dict[str, object] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "correlation_id": getattr(record, "correlation_id", "-"),
        }

        http_context = getattr(record, "request_context", None)
        if isinstance(http_context, RequestContext):
            base["http"] = {
                "method": http_context.method,
                "path": http_context.path,
                "user_agent": http_context.user_agent,
            }

        if record.exc_info:
            base["exception"] = self.formatException(record.exc_info)
        if record.stack_info:
            base["stack"] = self.formatStack(record.stack_info)

        # Attach any custom extras while avoiding double-counting standard attributes.
        standard_attrs = {
            "name",
            "msg",
            "args",
            "levelname",
            "levelno",
            "pathname",
            "filename",
            "module",
            "exc_info",
            "exc_text",
            "stack_info",
            "lineno",
            "funcName",
            "created",
            "msecs",
            "relativeCreated",
            "thread",
            "threadName",
            "processName",
            "process",
            "message",
            "correlation_id",
            "request_context",
        }
        extras: dict[str, object] = {}
        for key, value in record.__dict__.items():
            if key not in standard_attrs:
                extras[key] = value
        if extras:
            base["extra"] = extras

        return json.dumps(base, ensure_ascii=False)


def bind_correlation_id(value: Optional[str]) -> Token[Optional[str]]:
    """Bind ``value`` to the correlation id context variable."""

    return _correlation_id.set(value)


def reset_correlation_id(token: Token[Optional[str]]) -> None:
    """Reset the correlation id context variable to a previous state."""

    _correlation_id.reset(token)


def get_correlation_id() -> Optional[str]:
    """Return the current correlation id if bound."""

    return _correlation_id.get()


def bind_request_context(value: Optional[RequestContext]) -> Token[Optional[RequestContext]]:
    """Bind ``value`` to the request context variable."""

    return _request_context.set(value)


def reset_request_context(token: Token[Optional[RequestContext]]) -> None:
    """Reset the request context to a previous state."""

    _request_context.reset(token)


def get_request_context() -> Optional[RequestContext]:
    """Return the current request context if bound."""

    return _request_context.get()


@contextmanager
def correlation_id_context(value: Optional[str]) -> Iterator[None]:
    """Context manager that temporarily binds a correlation id."""

    token = bind_correlation_id(value)
    try:
        yield
    finally:
        reset_correlation_id(token)


@contextmanager
def request_context(value: Optional[RequestContext]) -> Iterator[None]:
    """Context manager that temporarily binds a request context."""

    token = bind_request_context(value)
    try:
        yield
    finally:
        reset_request_context(token)


def _ensure_handlers(logger: logging.Logger) -> None:
    if logger.handlers:
        return

    formatter = JsonLogFormatter()
    context_filter = ContextFilter()

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.addFilter(context_filter)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(LOG_FILE_PATH, mode="a", encoding="utf-8")
    file_handler.addFilter(context_filter)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger with correlation id filtering."""

    logger = logging.getLogger(name)
    logger.setLevel(LOG_LEVEL)
    _ensure_handlers(logger)
    return logger


__all__ = [
    "ContextFilter",
    "bind_correlation_id",
    "reset_correlation_id",
    "get_correlation_id",
    "correlation_id_context",
    "bind_request_context",
    "reset_request_context",
    "get_request_context",
    "request_context",
    "get_logger",
]
