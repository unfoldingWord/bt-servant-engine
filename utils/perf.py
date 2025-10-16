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
from dataclasses import dataclass, field
from typing import Optional, Any, Dict, List

from .pricing import get_pricing, get_pricing_details


# Current trace id propagated via ContextVar for async tasks
_current_trace_id: ContextVar[Optional[str]] = ContextVar("perf_current_trace_id", default=None)


@dataclass
class Span:
    """A single timed span with a name and timestamps."""

    name: str
    start: float
    end: float
    tokens: Dict[str, Optional[int]]
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


TIMING_MATCH_TOLERANCE = 1e-6

INTENT_NODE_MAP = {
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


@dataclass
class _OpenSpan:
    name: str
    start: float
    tokens: Dict[str, int] = field(default_factory=lambda: {
        "input": 0,
        "output": 0,
        "total": 0,
        "cached_input": 0,
        "audio_input": 0,
        "audio_output": 0,
    })
    model_token_breakdown: Dict[str, Dict[str, int]] | None = None


# Stack of open spans (per-task via ContextVar) to attribute tokens
_active_spans: ContextVar[List[_OpenSpan] | tuple] = ContextVar("perf_active_spans", default=())


@dataclass(slots=True)
class TokenIncrements:
    """Token deltas captured from model usage callbacks."""

    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    model: Optional[str] = None
    cached_input_tokens: Optional[int] = None
    audio_input_tokens: Optional[int] = None
    audio_output_tokens: Optional[int] = None


@dataclass(slots=True)
class TokenCounts:
    """Aggregate token counts for a span or grouped roll-up."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cached_input_tokens: int = 0
    audio_input_tokens: int = 0
    audio_output_tokens: int = 0


@dataclass(slots=True)
class CostSnapshot:
    """Cost totals (USD) attributed to a span or group."""

    input_cost_usd: float = 0.0
    output_cost_usd: float = 0.0
    cached_input_cost_usd: float = 0.0
    audio_input_cost_usd: float = 0.0
    audio_output_cost_usd: float = 0.0

    @property
    def total_cost_usd(self) -> float:
        """Return the combined cost of all token categories."""
        return (
            self.input_cost_usd
            + self.output_cost_usd
            + self.cached_input_cost_usd
            + self.audio_input_cost_usd
            + self.audio_output_cost_usd
        )


@dataclass(slots=True)
class AggregateTotals:
    """Roll-up of token, cost, and timing information for grouped spans."""

    tokens: TokenCounts = field(default_factory=TokenCounts)
    costs: CostSnapshot = field(default_factory=CostSnapshot)
    duration_ms: float = 0.0

    def add_counts(self, counts: TokenCounts) -> None:
        """Accumulate token totals into the grouped aggregate."""
        self.tokens.input_tokens += counts.input_tokens
        self.tokens.output_tokens += counts.output_tokens
        self.tokens.total_tokens += counts.total_tokens
        self.tokens.cached_input_tokens += counts.cached_input_tokens
        self.tokens.audio_input_tokens += counts.audio_input_tokens
        self.tokens.audio_output_tokens += counts.audio_output_tokens

    def add_costs(self, costs: CostSnapshot) -> None:
        """Accumulate cost totals into the grouped aggregate."""
        self.costs.input_cost_usd += costs.input_cost_usd
        self.costs.output_cost_usd += costs.output_cost_usd
        self.costs.cached_input_cost_usd += costs.cached_input_cost_usd
        self.costs.audio_input_cost_usd += costs.audio_input_cost_usd
        self.costs.audio_output_cost_usd += costs.audio_output_cost_usd


@dataclass(slots=True)
class TimingSummary:
    """Wall-clock timing metadata derived from collected spans."""

    start: float
    total_ms: float
    total_s: float

    @property
    def denominator(self) -> float:
        """Return a safe denominator to avoid division by zero."""
        return self.total_ms if self.total_ms > 0 else 1.0


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
        "cached_input": None,
        "audio_input": None,
        "audio_output": None,
    }
    mtb: Optional[Dict[str, Dict[str, int]]] = None
    if stack and stack[-1].name == name and abs(stack[-1].start - start) < TIMING_MATCH_TOLERANCE:
        os = stack[-1]
        current_tokens = os.tokens
        tokens["input"] = current_tokens.get("input") or None
        tokens["output"] = current_tokens.get("output") or None
        tokens["total"] = current_tokens.get("total") or None
        tokens["cached_input"] = current_tokens.get("cached_input") or None
        tokens["audio_input"] = current_tokens.get("audio_input") or None
        tokens["audio_output"] = current_tokens.get("audio_output") or None
        mtb = os.model_token_breakdown
    _store.add(
        tid,
        Span(
            name=name,
            start=start,
            end=end,
            tokens=tokens,
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


def _apply_token_deltas(open_span: _OpenSpan, increments: TokenIncrements) -> None:
    tokens = open_span.tokens
    if increments.input_tokens is not None:
        tokens["input"] += int(increments.input_tokens)
    if increments.output_tokens is not None:
        tokens["output"] += int(increments.output_tokens)
    if increments.total_tokens is not None:
        tokens["total"] += int(increments.total_tokens)
    elif increments.input_tokens is not None and increments.output_tokens is not None:
        tokens["total"] += int(increments.input_tokens) + int(increments.output_tokens)
    if increments.cached_input_tokens is not None:
        tokens["cached_input"] += int(increments.cached_input_tokens)
    if increments.audio_input_tokens is not None:
        tokens["audio_input"] += int(increments.audio_input_tokens)
    if increments.audio_output_tokens is not None:
        tokens["audio_output"] += int(increments.audio_output_tokens)


def _apply_model_breakdown(open_span: _OpenSpan, increments: TokenIncrements) -> None:
    if not increments.model:
        return
    if open_span.model_token_breakdown is None:
        open_span.model_token_breakdown = {}
    bucket = open_span.model_token_breakdown.setdefault(
        increments.model,
        {
            "input": 0,
            "output": 0,
            "total": 0,
            "cached_input": 0,
            "audio_input": 0,
            "audio_output": 0,
        },
    )
    if increments.input_tokens is not None:
        bucket["input"] += int(increments.input_tokens)
    if increments.output_tokens is not None:
        bucket["output"] += int(increments.output_tokens)
    if increments.total_tokens is not None:
        bucket["total"] += int(increments.total_tokens)
    elif increments.input_tokens is not None and increments.output_tokens is not None:
        bucket["total"] += int(increments.input_tokens) + int(increments.output_tokens)
    if increments.cached_input_tokens is not None:
        bucket["cached_input"] += int(increments.cached_input_tokens)
    if increments.audio_input_tokens is not None:
        bucket["audio_input"] += int(increments.audio_input_tokens)
    if increments.audio_output_tokens is not None:
        bucket["audio_output"] += int(increments.audio_output_tokens)


def add_tokens(increments: TokenIncrements) -> None:
    """Attach token counts to the current open span (if any)."""
    stack = _active_spans.get()
    if not stack:
        return
    open_span = stack[-1]
    _apply_token_deltas(open_span, increments)
    _apply_model_breakdown(open_span, increments)


def _sorted_spans(trace_id: str) -> list[Span]:
    return sorted(_store.get(trace_id), key=lambda span: span.start)


def _empty_summary(trace_id: str) -> Dict[str, Any]:
    return {"trace_id": trace_id, "total_ms": 0.0, "total_s": 0.0, "spans": []}


def _build_timing_summary(spans: list[Span]) -> TimingSummary:
    t0 = spans[0].start
    t1 = max(span.end for span in spans)
    total_ms = round((t1 - t0) * 1000.0, 2)
    total_s = round(t1 - t0, 2)
    return TimingSummary(start=t0, total_ms=total_ms, total_s=total_s)


def _token_denominator(spans: list[Span]) -> int:
    return sum(_span_token_counts(span).total_tokens for span in spans)


def _span_token_counts(span: Span) -> TokenCounts:
    bucket = span.tokens
    itok = bucket.get("input") or 0
    otok = bucket.get("output") or 0
    ttok = bucket.get("total") or (itok + otok)
    citok = bucket.get("cached_input") or 0
    aitok = bucket.get("audio_input") or 0
    aotok = bucket.get("audio_output") or 0
    return TokenCounts(
        input_tokens=itok,
        output_tokens=otok,
        total_tokens=ttok,
        cached_input_tokens=citok,
        audio_input_tokens=aitok,
        audio_output_tokens=aotok,
    )


def _calculate_costs(span: Span) -> CostSnapshot:
    if not span.model_token_breakdown:
        return CostSnapshot()

    costs = CostSnapshot()
    for model_name, tokens in span.model_token_breakdown.items():
        pricing = get_pricing(model_name)
        pricing_details = get_pricing_details(model_name)
        if not pricing:
            continue
        input_rate, output_rate = pricing
        costs.input_cost_usd += (tokens.get("input", 0) / 1_000_000.0) * input_rate
        costs.output_cost_usd += (tokens.get("output", 0) / 1_000_000.0) * output_rate
        if pricing_details and "cached_input_per_million" in pricing_details:
            costs.cached_input_cost_usd += (
                tokens.get("cached_input", 0) / 1_000_000.0
            ) * pricing_details["cached_input_per_million"]
        if pricing_details and "audio_input_per_million" in pricing_details:
            costs.audio_input_cost_usd += (
                tokens.get("audio_input", 0) / 1_000_000.0
            ) * pricing_details["audio_input_per_million"]
        if pricing_details and "audio_output_per_million" in pricing_details:
            costs.audio_output_cost_usd += (
                tokens.get("audio_output", 0) / 1_000_000.0
            ) * pricing_details["audio_output_per_million"]
    return costs


def _attach_costs(item: Dict[str, Any], costs: CostSnapshot) -> None:
    if not costs.total_cost_usd:
        return
    item["input_cost_usd"] = round(costs.input_cost_usd, 6)
    item["output_cost_usd"] = round(costs.output_cost_usd, 6)
    if costs.cached_input_cost_usd:
        item["cached_input_cost_usd"] = round(costs.cached_input_cost_usd, 6)
    if costs.audio_input_cost_usd:
        item["audio_input_cost_usd"] = round(costs.audio_input_cost_usd, 6)
    if costs.audio_output_cost_usd:
        item["audio_output_cost_usd"] = round(costs.audio_output_cost_usd, 6)
    item["total_cost_usd"] = round(costs.total_cost_usd, 6)


def _update_group_totals(
    grouped: Dict[str, AggregateTotals],
    span: Span,
    counts: TokenCounts,
    costs: CostSnapshot,
    duration_ms: float,
) -> None:
    if not span.name.startswith("brain:"):
        return
    node = span.name.split(":", 1)[1]
    intent = INTENT_NODE_MAP.get(node)
    if not intent:
        return
    aggregate = grouped.setdefault(intent, AggregateTotals())
    aggregate.add_counts(counts)
    aggregate.add_costs(costs)
    aggregate.duration_ms += duration_ms


def _token_percentage(total_tokens: int, denominator: int) -> str:
    denom = float(denominator) if denominator > 0 else 1.0
    return f"{round((total_tokens / denom) * 100.0, 1)}%"


def _format_grouped_totals(
    grouped: Dict[str, AggregateTotals],
    timing: TimingSummary,
    token_denominator: int,
) -> Dict[str, Dict[str, Any]]:
    total_ms = timing.total_ms or 0.0
    token_denom = token_denominator or 1
    result: Dict[str, Dict[str, Any]] = {}
    for intent, totals in grouped.items():
        duration_denom = total_ms if total_ms > 0 else 1.0
        tokens = totals.tokens
        costs = totals.costs
        result[intent] = {
            "input_tokens": int(tokens.input_tokens),
            "output_tokens": int(tokens.output_tokens),
            "total_tokens": int(tokens.total_tokens),
            "cached_input_tokens": int(tokens.cached_input_tokens),
            "audio_input_tokens": int(tokens.audio_input_tokens),
            "audio_output_tokens": int(tokens.audio_output_tokens),
            "input_cost_usd": round(costs.input_cost_usd, 6),
            "output_cost_usd": round(costs.output_cost_usd, 6),
            "cached_input_cost_usd": round(costs.cached_input_cost_usd, 6),
            "audio_input_cost_usd": round(costs.audio_input_cost_usd, 6),
            "audio_output_cost_usd": round(costs.audio_output_cost_usd, 6),
            "total_cost_usd": round(costs.total_cost_usd, 6),
            "duration_percentage": f"{round((totals.duration_ms / duration_denom) * 100.0, 1)}%",
            "token_percentage": _token_percentage(tokens.total_tokens, token_denom),
        }
    return result


@dataclass(slots=True)
class SummaryAssemblyContext:
    """Container of intermediate state while building a trace summary."""

    trace_id: str
    timing: TimingSummary
    totals: AggregateTotals
    grouped: Dict[str, AggregateTotals]
    items: list[Dict[str, Any]]
    token_denominator: int


def _span_summary(
    span: Span,
    timing: TimingSummary,
    token_denominator: int,
    totals: AggregateTotals,
    grouped: Dict[str, AggregateTotals],
) -> Dict[str, Any]:
    duration_seconds = span.end - span.start
    duration_ms = round(duration_seconds * 1000.0, 2)
    item: Dict[str, Any] = {
        "name": span.name,
        "duration_ms": duration_ms,
        "duration_se": round(duration_seconds, 2),
        "duration_percentage": f"{round((duration_ms / timing.denominator) * 100.0, 1)}%",
        "start_offset_ms": round((span.start - timing.start) * 1000.0, 2),
    }
    bucket = span.tokens
    if bucket.get("input") is not None:
        item["input_tokens_expended"] = bucket["input"]
    if bucket.get("output") is not None:
        item["output_tokens_expended"] = bucket["output"]
    if bucket.get("total") is not None:
        item["total_tokens_expended"] = bucket["total"]

    counts = _span_token_counts(span)
    costs = _calculate_costs(span)
    _attach_costs(item, costs)

    totals.add_counts(counts)
    totals.add_costs(costs)
    _update_group_totals(grouped, span, counts, costs, duration_ms)
    item["token_percentage"] = _token_percentage(counts.total_tokens, token_denominator)
    return item


def _assemble_summary(context: SummaryAssemblyContext) -> Dict[str, Any]:
    totals = context.totals
    timing = context.timing
    return {
        "trace_id": context.trace_id,
        "total_ms": timing.total_ms,
        "total_s": timing.total_s,
        "total_input_tokens": totals.tokens.input_tokens,
        "total_output_tokens": totals.tokens.output_tokens,
        "total_tokens": totals.tokens.total_tokens,
        "total_cached_input_tokens": totals.tokens.cached_input_tokens,
        "total_audio_input_tokens": totals.tokens.audio_input_tokens,
        "total_audio_output_tokens": totals.tokens.audio_output_tokens,
        "total_input_cost_usd": round(totals.costs.input_cost_usd, 6),
        "total_output_cost_usd": round(totals.costs.output_cost_usd, 6),
        "total_cached_input_cost_usd": round(totals.costs.cached_input_cost_usd, 6),
        "total_audio_input_cost_usd": round(totals.costs.audio_input_cost_usd, 6),
        "total_audio_output_cost_usd": round(totals.costs.audio_output_cost_usd, 6),
        "total_cost_usd": round(totals.costs.total_cost_usd, 6),
        "grouped_totals_by_intent": _format_grouped_totals(
            context.grouped, timing, context.token_denominator
        ),
        "spans": context.items,
    }


def summarize_report(trace_id: str) -> Dict[str, Any]:
    """Return an ordered summary of spans for the given trace id."""
    spans = _sorted_spans(trace_id)
    if not spans:
        return _empty_summary(trace_id)

    timing = _build_timing_summary(spans)
    token_denominator = _token_denominator(spans)

    totals = AggregateTotals()
    grouped: Dict[str, AggregateTotals] = {}
    items = [
        _span_summary(span, timing, token_denominator, totals, grouped) for span in spans
    ]

    context = SummaryAssemblyContext(
        trace_id=trace_id,
        timing=timing,
        totals=totals,
        grouped=grouped,
        items=items,
        token_denominator=token_denominator,
    )
    return _assemble_summary(context)


def log_final_report(logger: Any, trace_id: str, **metadata: Any) -> None:
    """Emit a single log.info line with the report and optional metadata.

    Cleans up stored spans for the trace afterwards to avoid memory growth.
    """
    report = summarize_report(trace_id)
    payload = {**metadata, **report}
    text = json.dumps(payload, separators=(",", ":"), indent=3)
    logger.info("PerfReport %s", text)
    _store.clear(trace_id)
