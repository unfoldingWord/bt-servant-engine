"""Custom FastAPI middleware components."""

from __future__ import annotations

import uuid
import time

from bt_servant_engine.core.logging import (
    bind_correlation_id,
    bind_request_context,
    get_logger,
    reset_correlation_id,
    reset_request_context,
)
from bt_servant_engine.core.models import RequestContext

logger = get_logger(__name__)


class CorrelationIdMiddleware:  # pylint: disable=too-few-public-methods
    """Bind a correlation id for each HTTP request and echo it in responses."""

    header_names = ("X-Request-ID", "X-Correlation-ID")

    def __init__(self, app) -> None:  # type: ignore[no-untyped-def]
        self.app = app

    async def __call__(self, scope, receive, send):  # type: ignore[no-untyped-def]
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        headers = {k.decode().lower(): v.decode() for k, v in scope.get("headers", [])}
        correlation_id = self._resolve_correlation_id(headers)
        token = bind_correlation_id(correlation_id)
        start_time = time.perf_counter()
        status_code: int | None = None
        state = scope.setdefault("state", {})
        request_ctx = RequestContext(
            correlation_id=correlation_id,
            path=scope.get("path", ""),
            method=scope.get("method", ""),
            user_agent=headers.get("user-agent"),
        )
        state["request_context"] = request_ctx
        request_token = bind_request_context(request_ctx)

        async def send_wrapper(message):  # type: ignore[no-untyped-def]
            nonlocal status_code
            if message.get("type") == "http.response.start":
                try:
                    status_code = int(message.get("status") or 0)
                except (TypeError, ValueError):
                    status_code = None
                header_list = list(message.get("headers", []))
                existing = {key.decode().lower() for key, _ in header_list}
                for header in self.header_names:
                    if header.lower() not in existing:
                        header_list.append((header.encode(), correlation_id.encode()))
                message["headers"] = header_list
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            logger.info(
                "request completed",
                extra={
                    "event": "http_request",
                    "status_code": status_code or 500,
                    "duration_ms": round(duration_ms, 2),
                },
            )
            reset_correlation_id(token)
            reset_request_context(request_token)

    def _resolve_correlation_id(self, header_map: dict[str, str]) -> str:
        for header in self.header_names:
            value = header_map.get(header.lower())
            if value:
                return value
        return uuid.uuid4().hex


__all__ = ["CorrelationIdMiddleware"]
