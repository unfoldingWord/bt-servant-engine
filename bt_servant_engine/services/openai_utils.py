"""Utilities for working with OpenAI SDK objects."""

from typing import Any, Callable, Optional

from utils.perf import add_tokens


def extract_cached_input_tokens(usage: Any) -> int | None:
    """Best-effort extraction of cached input token counts from SDK usage objects.

    Supports:
    - Responses API: usage.input_token_details.cache_read_input_tokens
    - Chat Completions: usage.prompt_tokens_details.cached_tokens

    Returns:
        Cached input token count if available, None otherwise.
    """
    try:
        itd = getattr(usage, "input_token_details", None)
        if itd is not None:
            val = getattr(itd, "cache_read_input_tokens", None)
            if val is None and isinstance(itd, dict):
                val = itd.get("cache_read_input_tokens")
            if isinstance(val, int) and val > 0:
                return val
        ptd = getattr(usage, "prompt_tokens_details", None)
        if ptd is not None:
            val2 = getattr(ptd, "cached_tokens", None)
            if val2 is None and isinstance(ptd, dict):
                val2 = ptd.get("cached_tokens")
            if isinstance(val2, int) and val2 > 0:
                return val2
    except Exception:  # pylint: disable=broad-except
        return None
    return None


def track_openai_usage(
    usage: Any,
    model: str,
    extract_cached_fn: Optional[Callable[[Any], Optional[int]]] = None,
    add_tokens_fn: Optional[Callable] = None,
) -> None:
    """Extract and track token usage from OpenAI response objects.

    Supports both Responses API (input_tokens/output_tokens) and
    Chat Completions API (prompt_tokens/completion_tokens).

    Args:
        usage: OpenAI usage object from response
        model: Model name for tracking
        extract_cached_fn: Optional function to extract cached tokens.
                          Defaults to extract_cached_input_tokens.
        add_tokens_fn: Optional function to add tokens for tracking.
                      Defaults to utils.perf.add_tokens.
    """
    if usage is None:
        return

    # Try Responses API field names first, then Chat Completions field names
    it = getattr(usage, "input_tokens", None) or getattr(usage, "prompt_tokens", None)
    ot = getattr(usage, "output_tokens", None) or getattr(usage, "completion_tokens", None)
    tt = getattr(usage, "total_tokens", None)

    # Calculate total if not provided
    if tt is None and (it is not None or ot is not None):
        tt = (it or 0) + (ot or 0)

    # Extract cached tokens
    cached_fn = extract_cached_fn or extract_cached_input_tokens
    cit = cached_fn(usage)

    # Add tokens to tracking
    tokens_fn = add_tokens_fn or add_tokens
    tokens_fn(it, ot, tt, model=model, cached_input_tokens=cit)


__all__ = ["extract_cached_input_tokens", "track_openai_usage"]
