"""Structured logging helpers with correlation, user, and client metadata."""

from __future__ import annotations

import logging
import sys
from contextlib import contextmanager
from contextvars import ContextVar, Token
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Iterator, Optional

from pythonjsonlogger import jsonlogger

from bt_servant_engine.core.config import settings

_correlation_id: ContextVar[Optional[str]] = ContextVar("correlation_id", default=None)
_log_user_id: ContextVar[Optional[str]] = ContextVar("log_user_id", default=None)
_client_ip: ContextVar[Optional[str]] = ContextVar("client_ip", default=None)

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
LOG_SCHEMA_VERSION = str(getattr(settings, "BT_SERVANT_LOG_SCHEMA_VERSION", "1.0.0"))


def _resolve_log_api_cutoff() -> datetime:
    """Normalize the minimum modified timestamp expected by admin log APIs."""

    configured = getattr(
        settings,
        "BT_SERVANT_LOG_API_MIN_MODIFIED_AT",
        datetime.now(timezone.utc),
    )
    if configured.tzinfo is None:
        return configured.replace(tzinfo=timezone.utc)
    return configured.astimezone(timezone.utc)


LOG_API_MIN_MODIFIED_AT = _resolve_log_api_cutoff()
LOG_FILE_PATH = LOGS_DIR / "bt_servant.log"
LOG_MAX_BYTES = 5 * 1024 * 1024
LOG_BACKUP_COUNT = 5


def _archive_legacy_log_if_needed() -> None:
    """Rename legacy logs that predate the configured cutoff."""
    cutoff = getattr(settings, "BT_SERVANT_LOG_ROTATE_BEFORE", None)
    if cutoff is None:
        return

    if cutoff.tzinfo is None:
        cutoff = cutoff.replace(tzinfo=timezone.utc)
    else:
        cutoff = cutoff.astimezone(timezone.utc)

    try:
        stat_result = LOG_FILE_PATH.stat()
    except FileNotFoundError:
        return
    except PermissionError:
        return
    except OSError:
        return

    modified_at = datetime.fromtimestamp(stat_result.st_mtime, tz=timezone.utc)
    if modified_at >= cutoff:
        return

    archive_path = LOG_FILE_PATH.with_name(f"{LOG_FILE_PATH.name}.old")
    if archive_path.exists():
        timestamp_suffix = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        archive_path = archive_path.with_name(f"{archive_path.name}.{timestamp_suffix}")

    try:
        LOG_FILE_PATH.rename(archive_path)
    except OSError:
        return


_archive_legacy_log_if_needed()


class VersionedJsonFormatter(jsonlogger.JsonFormatter):
    """Inject a schema version into each structured log entry."""

    def __init__(self, *args, schema_version: str, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._schema_version = schema_version

    def add_fields(
        self,
        log_record: dict[str, Any],
        record: logging.LogRecord,
        message_dict: dict[str, Any],
    ) -> None:
        super().add_fields(log_record, record, message_dict)
        # Preserve explicit overrides on the record.
        # Otherwise ensure every entry includes the schema version.
        log_record.setdefault("schema_version", self._schema_version)


class CorrelationIdFilter(logging.Filter):  # pylint: disable=too-few-public-methods
    """Attach correlation, user, and client metadata to log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.correlation_id = get_correlation_id() or "-"
        record.log_user_id = get_log_user_id() or "-"
        record.client_ip = get_client_ip() or "-"
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


def bind_log_user_id(value: Optional[str]) -> Token[Optional[str]]:
    """Bind the pseudonymized user identifier for downstream logging."""

    return _log_user_id.set(value)


def reset_log_user_id(token: Token[Optional[str]]) -> None:
    """Reset the pseudonymized user identifier context variable."""

    _log_user_id.reset(token)


def get_log_user_id() -> Optional[str]:
    """Return the current pseudonymized user identifier if bound."""

    return _log_user_id.get()


def bind_client_ip(value: Optional[str]) -> Token[Optional[str]]:
    """Bind the client IP address associated with the current request."""

    return _client_ip.set(value)


def reset_client_ip(token: Token[Optional[str]]) -> None:
    """Reset the client IP context variable."""

    _client_ip.reset(token)


def get_client_ip() -> Optional[str]:
    """Return the current client IP if bound."""

    return _client_ip.get()


@contextmanager
def correlation_id_context(value: Optional[str]) -> Iterator[None]:
    """Context manager that temporarily binds a correlation id."""

    token = bind_correlation_id(value)
    try:
        yield
    finally:
        reset_correlation_id(token)


@contextmanager
def log_user_id_context(value: Optional[str]) -> Iterator[None]:
    """Context manager that temporarily binds a pseudonymized user id."""

    token = bind_log_user_id(value)
    try:
        yield
    finally:
        reset_log_user_id(token)


@contextmanager
def client_ip_context(value: Optional[str]) -> Iterator[None]:
    """Context manager that temporarily binds a client IP address."""

    token = bind_client_ip(value)
    try:
        yield
    finally:
        reset_client_ip(token)


def _ensure_handlers(logger: logging.Logger) -> None:
    if logger.handlers:
        return

    formatter = VersionedJsonFormatter(
        " ".join(
            [
                "%(asctime)s",
                "%(levelname)s",
                "%(name)s",
                "%(message)s",
                "%(correlation_id)s",
                "%(log_user_id)s",
                "%(client_ip)s",
            ]
        ),
        rename_fields={
            "asctime": "timestamp",
            "levelname": "level",
            "name": "logger",
            "correlation_id": "cid",
            "log_user_id": "user",
        },
        datefmt="%Y-%m-%d %H:%M:%S",
        json_ensure_ascii=False,
        schema_version=LOG_SCHEMA_VERSION,
    )
    correlation_filter = CorrelationIdFilter()

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.addFilter(correlation_filter)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = RotatingFileHandler(
        LOG_FILE_PATH, maxBytes=LOG_MAX_BYTES, backupCount=LOG_BACKUP_COUNT, encoding="utf-8"
    )
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
    "bind_client_ip",
    "bind_log_user_id",
    "reset_correlation_id",
    "reset_client_ip",
    "reset_log_user_id",
    "get_correlation_id",
    "get_client_ip",
    "get_log_user_id",
    "correlation_id_context",
    "client_ip_context",
    "log_user_id_context",
    "get_logger",
    "LOG_SCHEMA_VERSION",
    "LOG_API_MIN_MODIFIED_AT",
]
