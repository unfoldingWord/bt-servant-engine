"""Lightweight performance tracing utilities for per-message timing.

Provides a thread- and async-safe way to record spans keyed by a trace id
(we use WhatsApp `message_id`). Designed to be non-invasive and avoid
behavior changes while giving a final per-trace report.
"""
from __future__ import annotations

import time
import json
import asyncio
import threading
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Optional, Any, Dict, List


# Current trace id propagated via ContextVar for async tasks
_current_trace_id: ContextVar[Optional[str]] = ContextVar("perf_current_trace_id", default=None)


@dataclass
class Span:  # pylint: disable=too-many-instance-attributes
    """A single timed span with a name and timestamps."""
    name: str
    start: float
    end: float
    input_tokens_expended: Optional[int] = None
    output_tokens_expended: Optional[int] = None
    total_tokens_expended: Optional[int] = None
    cached_input_tokens_expended: Optional[int] = None
    # Audio token accounting (optional, when SDK provides counts or we estimate)
    audio_input_tokens_expended: Optional[int] = None
    audio_output_tokens_expended: Optional[int] = None
    # Optional model-level token breakdown:
    # { model: {"input": int, "output": int, "total": int, "cached_input": int,
    #           "audio_input": int, "audio_output": int} }
    model_token_breakdown: Optional[Dict[str, Dict[str, int]]] = None

    @property
    def duration_ms(self) -> float:
        """Return the span duration in milliseconds."""
        return (self.end - self.start) * 1000.0


class _TraceStore:
    """In-memory store for spans keyed by trace id."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._spans: Dict[str, List[Span]] = {}

    def add(self, trace_id: str, span: Span) -> None:
        """Append a span to the trace's list."""
        with self._lock:
            self._spans.setdefault(trace_id, []).append(span)

    def get(self, trace_id: str) -> List[Span]:
        """Return a shallow copy of spans for the trace id."""
        with self._lock:
            return list(self._spans.get(trace_id, []))

    def clear(self, trace_id: str) -> None:
        """Clear all spans for the given trace id."""
        with self._lock:
            if trace_id in self._spans:
                del self._spans[trace_id]


_store = _TraceStore()


@dataclass
class _OpenSpan:  # pylint: disable=too-many-instance-attributes
    name: str
    start: float
    input_tokens_expended: int = 0
    output_tokens_expended: int = 0
    total_tokens_expended: int = 0
    cached_input_tokens_expended: int = 0
    audio_input_tokens_expended: int = 0
    audio_output_tokens_expended: int = 0
    model_token_breakdown: Dict[str, Dict[str, int]] | None = None


# Stack of open spans (per-task via ContextVar) to attribute tokens
_active_spans: ContextVar[List[_OpenSpan] | tuple] = ContextVar("perf_active_spans", default=())


def set_current_trace(trace_id: Optional[str]) -> None:
    """Set the ContextVar-based current trace id for downstream spans."""
    _current_trace_id.set(trace_id)


def get_current_trace() -> Optional[str]:
    """Return the current ContextVar-based trace id if set."""
    return _current_trace_id.get()


def _record_span(name: str, start: float, end: float, trace_id: Optional[str] = None) -> None:
    tid = trace_id or get_current_trace()
    if not tid:
        return
    # If there is an active open span matching this name at the top, merge token totals
    stack = _active_spans.get()
    tokens: Dict[str, Optional[int]] = {
        "input": None,
        "output": None,
        "total": None,
        "audio_input": None,
        "audio_output": None,
    }
    mtb: Optional[Dict[str, Dict[str, int]]] = None
    cached_input = None
    if stack and stack[-1].name == name and abs(stack[-1].start - start) < 1e-6:
        os = stack[-1]
        # Normalize zero totals to None when no tokens were added at all
        tokens["input"] = os.input_tokens_expended or None
        tokens["output"] = os.output_tokens_expended or None
        tokens["total"] = os.total_tokens_expended or None
        cached_input = os.cached_input_tokens_expended or None
        tokens["audio_input"] = os.audio_input_tokens_expended or None
        tokens["audio_output"] = os.audio_output_tokens_expended or None
        mtb = os.model_token_breakdown
    _store.add(
        tid,
        Span(
            name=name,
            start=start,
            end=end,
            input_tokens_expended=tokens["input"],
            output_tokens_expended=tokens["output"],
            total_tokens_expended=tokens["total"],
            cached_input_tokens_expended=cached_input,
            audio_input_tokens_expended=tokens["audio_input"],
            audio_output_tokens_expended=tokens["audio_output"],
            model_token_breakdown=mtb,
        ),
    )


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
        # Push open span to the stack
        cur = _active_spans.get()
        stack = list(cur)  # default may be tuple()
        stack.append(_OpenSpan(name=self.name, start=self._start))
        _active_spans.set(stack)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._end = time.time()
        # Record before popping so tokens are visible to recorder
        _record_span(self.name, self._start, self._end, self.trace_id)
        stack = list(_active_spans.get())
        if stack:
            stack.pop()
            _active_spans.set(stack)

    # async protocol
    async def __aenter__(self) -> "PerfBlock":
        self._start = time.time()
        cur = _active_spans.get()
        stack = list(cur)
        stack.append(_OpenSpan(name=self.name, start=self._start))
        _active_spans.set(stack)
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        self._end = time.time()
        _record_span(self.name, self._start, self._end, self.trace_id)
        stack = list(_active_spans.get())
        if stack:
            stack.pop()
            _active_spans.set(stack)


def time_block(name: str, trace_id: Optional[str] = None) -> PerfBlock:
    """Create a timing block for `with` or `async with` usage."""
    return PerfBlock(name, trace_id=trace_id)


def record_timing(name: str):
    """Decorator to record sync or async function execution time as a span."""
    def decorator(func):  # type: ignore[no-untyped-def]
        is_async = asyncio.iscoroutinefunction(func)

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


def record_external_span(
    name: str, start: float, end: float, trace_id: Optional[str] = None
) -> None:
    """Record a span using externally measured timestamps."""
    _record_span(name, start, end, trace_id=trace_id)


def add_tokens(  # pylint: disable=too-many-branches,too-many-arguments
    input_tokens: Optional[int] = None,
    output_tokens: Optional[int] = None,
    total_tokens: Optional[int] = None,
    *,
    model: Optional[str] = None,
    cached_input_tokens: Optional[int] = None,
    audio_input_tokens: Optional[int] = None,
    audio_output_tokens: Optional[int] = None,
) -> None:
    """Attach token counts to the current open span (if any).

    - Sums values if called multiple times within the same span.
    - If ``total_tokens`` is not provided, a sum of available inputs is computed
      when both input and output tokens are present.
    """
    stack = _active_spans.get()
    if not stack:
        return
    os = stack[-1]
    if input_tokens is not None:
        os.input_tokens_expended += int(input_tokens)
    if output_tokens is not None:
        os.output_tokens_expended += int(output_tokens)
    if total_tokens is not None:
        os.total_tokens_expended += int(total_tokens)
    elif input_tokens is not None or output_tokens is not None:
        # Estimate total for this call if both parts available; otherwise leave
        # accumulation to future calls when the missing part arrives.
        if input_tokens is not None and output_tokens is not None:
            os.total_tokens_expended += int(input_tokens) + int(output_tokens)
    # Track cached input tokens
    if cached_input_tokens is not None:
        os.cached_input_tokens_expended += int(cached_input_tokens)

    # Track audio tokens (separately from text tokens)
    if audio_input_tokens is not None:
        os.audio_input_tokens_expended += int(audio_input_tokens)
    if audio_output_tokens is not None:
        os.audio_output_tokens_expended += int(audio_output_tokens)

    # Track per-model breakdown
    if model:
        if os.model_token_breakdown is None:
            os.model_token_breakdown = {}
        bucket = os.model_token_breakdown.setdefault(
            model,
            {
                "input": 0,
                "output": 0,
                "total": 0,
                "cached_input": 0,
                "audio_input": 0,
                "audio_output": 0,
            },
        )
        if input_tokens is not None:
            bucket["input"] += int(input_tokens)
        if output_tokens is not None:
            bucket["output"] += int(output_tokens)
        if total_tokens is not None:
            bucket["total"] += int(total_tokens)
        elif input_tokens is not None and output_tokens is not None:
            bucket["total"] += int(input_tokens) + int(output_tokens)
        if cached_input_tokens is not None:
            bucket["cached_input"] += int(cached_input_tokens)
        if audio_input_tokens is not None:
            bucket["audio_input"] += int(audio_input_tokens)
        if audio_output_tokens is not None:
            bucket["audio_output"] += int(audio_output_tokens)


def summarize_report(trace_id: str) -> Dict[str, Any]:  # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    """Return an ordered summary of spans for the given trace id.

    Adds both millisecond and second totals, and augments each span with
    seconds and a duration_percentage (duration_ms / total_ms).
    """
    spans = sorted(_store.get(trace_id), key=lambda s: s.start)
    if not spans:
        return {"trace_id": trace_id, "total_ms": 0.0, "total_s": 0.0, "spans": []}

    t0 = spans[0].start
    t1 = max(s.end for s in spans)
    total_ms = round((t1 - t0) * 1000.0, 2)
    total_s = round((t1 - t0), 2)

    # Guard against divide-by-zero if timestamps are identical
    denom = total_ms if total_ms > 0 else 1.0

    from .pricing import get_pricing, get_pricing_details  # pylint: disable=import-outside-toplevel

    # Precompute total token denominator for token_percentage.
    token_total_denominator = 0
    for _s in spans:
        _itok = _s.input_tokens_expended or 0
        _otok = _s.output_tokens_expended or 0
        _ttok = _s.total_tokens_expended or (_itok + _otok)
        token_total_denominator += _ttok

    items: List[Dict[str, Any]] = []
    total_input_tokens = 0
    total_output_tokens = 0
    total_tokens = 0
    total_cost_input_usd = 0.0
    total_cost_output_usd = 0.0
    total_cost_cached_input_usd = 0.0
    total_cost_usd = 0.0
    total_cached_input_tokens = 0
    total_audio_input_tokens = 0
    total_audio_output_tokens = 0
    total_cost_audio_input_usd = 0.0
    total_cost_audio_output_usd = 0.0

    # Intent grouping: map node names to intent identifiers
    intent_node_map: Dict[str, str] = {
        "query_vector_db_node": "get-bible-translation-assistance",
        "query_open_ai_node": "get-bible-translation-assistance",
        "handle_get_passage_summary_node": "get-passage-summary",
        "handle_get_passage_keywords_node": "get-passage-keywords",
        "handle_get_translation_helps_node": "get-translation-helps",
        "set_response_language_node": "set-response-language",
        "handle_unsupported_function_node": "perform-unsupported-function",
        "handle_system_information_request_node": "retrieve-system-information",
        "converse_with_bt_servant_node": "converse-with-bt-servant",
    }
    grouped: Dict[str, Dict[str, float]] = {}
    for s in spans:
        dur_ms = round((s.end - s.start) * 1000.0, 2)
        item: Dict[str, Any] = {
            "name": s.name,
            "duration_ms": dur_ms,
            "duration_se": round((s.end - s.start), 2),
            "duration_percentage": f"{round((dur_ms / denom) * 100.0, 1)}%",
            "start_offset_ms": round((s.start - t0) * 1000.0, 2),
        }
        if s.input_tokens_expended is not None:
            item["input_tokens_expended"] = s.input_tokens_expended
        if s.output_tokens_expended is not None:
            item["output_tokens_expended"] = s.output_tokens_expended
        if s.total_tokens_expended is not None:
            item["total_tokens_expended"] = s.total_tokens_expended
        # Compute cost per span using model breakdown when available
        span_cost_input = 0.0
        span_cost_output = 0.0
        span_cost_cached_input = 0.0
        span_cost_total = 0.0
        span_cost_audio_input = 0.0
        span_cost_audio_output = 0.0
        if s.model_token_breakdown:
            for model_name, tok in s.model_token_breakdown.items():
                pricing = get_pricing(model_name)
                pricing_details = get_pricing_details(model_name)
                if not pricing:
                    continue
                in_price, out_price = pricing
                span_cost_input += (tok.get("input", 0) / 1_000_000.0) * in_price
                span_cost_output += (tok.get("output", 0) / 1_000_000.0) * out_price
                if pricing_details and "cached_input_per_million" in pricing_details:
                    span_cost_cached_input += (
                        (tok.get("cached_input", 0) / 1_000_000.0)
                        * pricing_details["cached_input_per_million"]
                    )
                if pricing_details and "audio_input_per_million" in pricing_details:
                    span_cost_audio_input += (
                        (tok.get("audio_input", 0) / 1_000_000.0)
                        * pricing_details["audio_input_per_million"]
                    )
                if pricing_details and "audio_output_per_million" in pricing_details:
                    span_cost_audio_output += (
                        (tok.get("audio_output", 0) / 1_000_000.0)
                        * pricing_details["audio_output_per_million"]
                    )
        # If no per-model breakdown is available, we skip cost for this span.
        # Pricing requires a model to resolve input/output rates.
        if (
            span_cost_input
            or span_cost_output
            or span_cost_cached_input
            or span_cost_audio_input
            or span_cost_audio_output
        ):
            span_cost_total = (
                span_cost_input
                + span_cost_output
                + span_cost_cached_input
                + span_cost_audio_input
                + span_cost_audio_output
            )
            item["input_cost_usd"] = round(span_cost_input, 6)
            item["output_cost_usd"] = round(span_cost_output, 6)
            if span_cost_cached_input:
                item["cached_input_cost_usd"] = round(span_cost_cached_input, 6)
            if span_cost_audio_input:
                item["audio_input_cost_usd"] = round(span_cost_audio_input, 6)
            if span_cost_audio_output:
                item["audio_output_cost_usd"] = round(span_cost_audio_output, 6)
            item["total_cost_usd"] = round(span_cost_total, 6)

        # Update top-level totals
        itok = s.input_tokens_expended or 0
        otok = s.output_tokens_expended or 0
        ttok = s.total_tokens_expended or (itok + otok)
        citok = s.cached_input_tokens_expended or 0
        aitok = s.audio_input_tokens_expended or 0
        aotok = s.audio_output_tokens_expended or 0
        total_input_tokens += itok
        total_output_tokens += otok
        total_tokens += ttok
        total_cached_input_tokens += citok
        total_audio_input_tokens += aitok
        total_audio_output_tokens += aotok
        total_cost_input_usd += span_cost_input
        total_cost_output_usd += span_cost_output
        total_cost_cached_input_usd += span_cost_cached_input
        total_cost_audio_input_usd += span_cost_audio_input
        total_cost_audio_output_usd += span_cost_audio_output
        total_cost_usd += span_cost_total

        # Token percentage: share of total tokens in this trace.
        token_den = float(token_total_denominator) if token_total_denominator > 0 else 1.0
        item["token_percentage"] = f"{round((ttok / token_den) * 100.0, 1)}%"

        # Group by intent when the span name matches a known intent node
        if s.name.startswith("brain:"):
            node = s.name.split(":", 1)[1]
            intent = intent_node_map.get(node)
            if intent:
                agg = grouped.setdefault(intent, {
                    "input_tokens": 0.0,
                    "output_tokens": 0.0,
                    "total_tokens": 0.0,
                    "cached_input_tokens": 0.0,
                    "audio_input_tokens": 0.0,
                    "audio_output_tokens": 0.0,
                    "input_cost_usd": 0.0,
                    "output_cost_usd": 0.0,
                    "cached_input_cost_usd": 0.0,
                    "audio_input_cost_usd": 0.0,
                    "audio_output_cost_usd": 0.0,
                    "total_cost_usd": 0.0,
                    "duration_ms": 0.0,
                })
                agg["input_tokens"] += itok
                agg["output_tokens"] += otok
                agg["total_tokens"] += ttok
                agg["cached_input_tokens"] += citok
                agg["audio_input_tokens"] += aitok
                agg["audio_output_tokens"] += aotok
                agg["input_cost_usd"] += span_cost_input
                agg["output_cost_usd"] += span_cost_output
                agg["cached_input_cost_usd"] += span_cost_cached_input
                agg["audio_input_cost_usd"] += span_cost_audio_input
                agg["audio_output_cost_usd"] += span_cost_audio_output
                agg["total_cost_usd"] += span_cost_total
                agg["duration_ms"] += dur_ms

        items.append(item)

    return {
        "trace_id": trace_id,
        "total_ms": total_ms,
        "total_s": total_s,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_tokens": total_tokens,
        "total_cached_input_tokens": total_cached_input_tokens,
        "total_audio_input_tokens": total_audio_input_tokens,
        "total_audio_output_tokens": total_audio_output_tokens,
        "total_input_cost_usd": round(total_cost_input_usd, 6),
        "total_output_cost_usd": round(total_cost_output_usd, 6),
        "total_cached_input_cost_usd": round(total_cost_cached_input_usd, 6),
        "total_audio_input_cost_usd": round(total_cost_audio_input_usd, 6),
        "total_audio_output_cost_usd": round(total_cost_audio_output_usd, 6),
        "total_cost_usd": round(total_cost_usd, 6),
        "grouped_totals_by_intent": {k: {
            "input_tokens": int(v["input_tokens"]),
            "output_tokens": int(v["output_tokens"]),
            "total_tokens": int(v["total_tokens"]),
            "cached_input_tokens": int(v["cached_input_tokens"]),
            "audio_input_tokens": int(v["audio_input_tokens"]),
            "audio_output_tokens": int(v["audio_output_tokens"]),
            "input_cost_usd": round(v["input_cost_usd"], 6),
            "output_cost_usd": round(v["output_cost_usd"], 6),
            "cached_input_cost_usd": round(v["cached_input_cost_usd"], 6),
            "audio_input_cost_usd": round(v["audio_input_cost_usd"], 6),
            "audio_output_cost_usd": round(v["audio_output_cost_usd"], 6),
            "total_cost_usd": round(v["total_cost_usd"], 6),
            "duration_percentage": (
                f"{round((
                    (v.get('duration_ms', 0.0) or 0.0)
                    / (total_ms or 1.0)
                ) * 100.0, 1)}%"
            ),
            "token_percentage": (
                f"{round((
                    (v.get('total_tokens', 0.0) or 0.0)
                    / (token_total_denominator or 1.0)
                ) * 100.0, 1)}%"
            ),
        } for k, v in grouped.items()},
        "spans": items,
    }


def log_final_report(logger: Any, trace_id: str, **metadata: Any) -> None:
    """Emit a single log.info line with the report and optional metadata.

    Cleans up stored spans for the trace afterwards to avoid memory growth.
    """
    report = summarize_report(trace_id)
    payload = {**metadata, **report}
    text = json.dumps(payload, separators=(",", ":"), indent=3)
    logger.info("PerfReport %s", text)
    _store.clear(trace_id)
