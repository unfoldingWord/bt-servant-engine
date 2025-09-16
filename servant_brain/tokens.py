"""Token accounting helpers shared by brain nodes."""
from __future__ import annotations

from typing import Any


def extract_cached_input_tokens(usage: Any) -> int | None:
    """Best-effort extraction of cached input token counts from SDK usage objects."""
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


__all__ = ["extract_cached_input_tokens"]
