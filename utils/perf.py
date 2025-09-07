"""Lightweight performance tracing utilities for per-message timing.

Provides a thread- and async-safe way to record spans keyed by a trace id
(we use WhatsApp `message_id`). Designed to be non-invasive and avoid
behavior changes while giving a final per-trace report.
"""
from __future__ import annotations

import time
import threading
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Optional, Any, Dict, List


# Current trace id propagated via ContextVar for async tasks
_current_trace_id: ContextVar[Optional[str]] = ContextVar("perf_current_trace_id", default=None)


@dataclass
class Span:
    name: str
    start: float
    end: float

    @property
    def duration_ms(self) -> float:
        return (self.end - self.start) * 1000.0


class _TraceStore:
    """In-memory store for spans keyed by trace id."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._spans: Dict[str, List[Span]] = {}

    def add(self, trace_id: str, span: Span) -> None:
        with self._lock:
            self._spans.setdefault(trace_id, []).append(span)

    def get(self, trace_id: str) -> List[Span]:
        with self._lock:
            return list(self._spans.get(trace_id, []))

    def clear(self, trace_id: str) -> None:
        with self._lock:
            if trace_id in self._spans:
                del self._spans


_store = _TraceStore()


def set_current_trace(trace_id: Optional[str]) -> None:
    """Set the ContextVar-based current trace id for downstream spans."""
    _current_trace_id.set(trace_id)


def get_current_trace() -> Optional[str]:
    return _current_trace_id.get()


def _record_span(name: str, start: float, end: float, trace_id: Optional[str] = None) -> None:
    tid = trace_id or get_current_trace()
    if not tid:
        return
    _store.add(tid, Span(name=name, start=start, end=end))


class PerfBlock:
    """A context manager (sync or async) to record a named span."""

    def __init__(self, name: str, trace_id: Optional[str] = None) -> None:
        self.name = name
        self.trace_id = trace_id
        self._start = 0.0
        self._end = 0.0

    # sync protocol
    def __enter__(self) -> "PerfBlock":
        self._start = time.time()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001 - follow context manager protocol
        self._end = time.time()
        _record_span(self.name, self._start, self._end, self.trace_id)

    # async protocol
    async def __aenter__(self) -> "PerfBlock":
        self._start = time.time()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001 - follow context manager protocol
        self._end = time.time()
        _record_span(self.name, self._start, self._end, self.trace_id)


def time_block(name: str, trace_id: Optional[str] = None) -> PerfBlock:
    """Create a timing block for `with` or `async with` usage."""
    return PerfBlock(name, trace_id=trace_id)


def record_timing(name: str):
    """Decorator to record sync or async function execution time as a span."""
    def decorator(func):  # type: ignore[no-untyped-def]
        try:
            import asyncio as _asyncio  # local import to avoid unconditional dependency
            is_async = _asyncio.iscoroutinefunction(func)
        except Exception:
            is_async = False

        if is_async:
            async def wrapper(*args, **kwargs):  # type: ignore[no-untyped-def]
                start = time.time()
                try:
                    return await func(*args, **kwargs)
                finally:
                    _record_span(name, start, time.time())
            return wrapper

        def wrapper_sync(*args, **kwargs):  # type: ignore[no-untyped-def]
            start = time.time()
            try:
                return func(*args, **kwargs)
            finally:
                _record_span(name, start, time.time())
        return wrapper_sync
    return decorator


def record_external_span(name: str, start: float, end: float, trace_id: Optional[str] = None) -> None:
    """Record a span using externally measured timestamps."""
    _record_span(name, start, end, trace_id=trace_id)


def summarize_report(trace_id: str) -> Dict[str, Any]:
    """Return an ordered summary of spans for the given trace id."""
    spans = sorted(_store.get(trace_id), key=lambda s: s.start)
    if not spans:
        return {"trace_id": trace_id, "total_ms": 0.0, "spans": []}
    t0 = spans[0].start
    t1 = max(s.end for s in spans)
    items = [
        {
            "name": s.name,
            "duration_ms": round((s.end - s.start) * 1000.0, 2),
            "start_offset_ms": round((s.start - t0) * 1000.0, 2),
        }
        for s in spans
    ]
    return {
        "trace_id": trace_id,
        "total_ms": round((t1 - t0) * 1000.0, 2),
        "spans": items,
    }


def log_final_report(logger: Any, trace_id: str, **metadata: Any) -> None:  # noqa: ANN401 - logger type varies
    """Emit a single log.info line with the report and optional metadata.

    Cleans up stored spans for the trace afterwards to avoid memory growth.
    """
    try:
        import json as _json
    except Exception:  # pragma: no cover - json always present but guard anyway
        _json = None  # type: ignore[assignment]

    report = summarize_report(trace_id)
    payload = {**metadata, **report}
    if _json is not None:
        text = _json.dumps(payload, separators=(",", ":"))
    else:
        text = str(payload)
    logger.info("PerfReport %s", text)
    _store.clear(trace_id)
