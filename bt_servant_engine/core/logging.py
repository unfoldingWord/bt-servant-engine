"""Structured logging helpers with correlation id support."""

from __future__ import annotations

import logging
import sys
from contextlib import contextmanager
from contextvars import ContextVar, Token
from pathlib import Path
from typing import Iterator, Optional

from bt_servant_engine.core.config import settings

_correlation_id: ContextVar[Optional[str]] = ContextVar("correlation_id", default=None)

LEVEL_NAME = str(getattr(settings, "BT_SERVANT_LOG_LEVEL", "info")).upper()
LOG_LEVEL = getattr(logging, LEVEL_NAME, logging.INFO)

BASE_DIR = Path(__file__).resolve().parent.parent
ROOT_DIR = BASE_DIR.parent


def _resolve_logs_dir() -> Path:
    """Select a writable logs directory honoring configuration overrides."""

    configured_dir = getattr(settings, "BT_SERVANT_LOG_DIR", None)
    candidates = []
    if configured_dir:
        candidates.append(Path(configured_dir))

    data_dir = Path(getattr(settings, "DATA_DIR", Path("/data")))
    # Precedence: explicit override → repo root logs → DATA_DIR/logs → package-local logs
    candidates.append(ROOT_DIR / "logs")
    candidates.append(data_dir / "logs")
    candidates.append(BASE_DIR / "logs")

    for candidate in candidates:
        try:
            candidate.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            continue
        return candidate

    raise PermissionError("Unable to create a writable logs directory")


LOGS_DIR = _resolve_logs_dir()
LOG_FILE_PATH = LOGS_DIR / "bt_servant.log"


class CorrelationIdFilter(logging.Filter):  # pylint: disable=too-few-public-methods
    """Attach the current correlation id to log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.correlation_id = get_correlation_id() or "-"
        return True


def bind_correlation_id(value: Optional[str]) -> Token[Optional[str]]:
    """Bind ``value`` to the correlation id context variable."""

    return _correlation_id.set(value)


def reset_correlation_id(token: Token[Optional[str]]) -> None:
    """Reset the correlation id context variable to a previous state."""

    _correlation_id.reset(token)


def get_correlation_id() -> Optional[str]:
    """Return the current correlation id if bound."""

    return _correlation_id.get()


@contextmanager
def correlation_id_context(value: Optional[str]) -> Iterator[None]:
    """Context manager that temporarily binds a correlation id."""

    token = bind_correlation_id(value)
    try:
        yield
    finally:
        reset_correlation_id(token)


def _ensure_handlers(logger: logging.Logger) -> None:
    if logger.handlers:
        return

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] [cid=%(correlation_id)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    correlation_filter = CorrelationIdFilter()

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.addFilter(correlation_filter)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(LOG_FILE_PATH, mode="a", encoding="utf-8")
    file_handler.addFilter(correlation_filter)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger with correlation id filtering."""

    logger = logging.getLogger(name)
    logger.setLevel(LOG_LEVEL)
    _ensure_handlers(logger)
    return logger


__all__ = [
    "CorrelationIdFilter",
    "bind_correlation_id",
    "reset_correlation_id",
    "get_correlation_id",
    "correlation_id_context",
    "get_logger",
]
