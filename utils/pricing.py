"""Pricing table and helper for model token costs.

By default, no pricing is configured. You can provide pricing via the
`OPENAI_PRICING_JSON` env var containing a JSON object like:

{
  "gpt-4o": {"input_per_million": 5.0, "output_per_million": 15.0},
  "gpt-4o-mini": {"input_per_million": 0.15, "output_per_million": 0.6}
}

If pricing is not provided for a model, cost fields will be omitted in
the perf reports for spans using that model.
"""
from __future__ import annotations

import json
import os
from typing import Optional, Tuple, Dict

_PRICING_CACHE: Optional[Dict[str, Dict[str, float]]] = None


def _load_pricing() -> Dict[str, Dict[str, float]]:
    global _PRICING_CACHE  # pylint: disable=global-statement
    if _PRICING_CACHE is not None:
        return _PRICING_CACHE
    raw = os.environ.get("OPENAI_PRICING_JSON", "").strip()
    table: Dict[str, Dict[str, float]] = {}
    if not raw:
        _PRICING_CACHE = table
        return table
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError, ValueError):
        data = {}
    if isinstance(data, dict):
        for model, cfg in data.items():
            if not isinstance(cfg, dict):
                continue
            inp = cfg.get("input_per_million")
            out = cfg.get("output_per_million")
            cached_in = cfg.get("cached_input") or cfg.get("cached_input_per_million")
            ain = cfg.get("audio_input_per_million") or cfg.get("audio_input")
            aout = cfg.get("audio_output_per_million") or cfg.get("audio_output")
            if isinstance(inp, (int, float)) and isinstance(out, (int, float)):
                entry: Dict[str, float] = {
                    "input_per_million": float(inp),
                    "output_per_million": float(out),
                }
                if isinstance(cached_in, (int, float)):
                    entry["cached_input_per_million"] = float(cached_in)
                if isinstance(ain, (int, float)):
                    entry["audio_input_per_million"] = float(ain)
                if isinstance(aout, (int, float)):
                    entry["audio_output_per_million"] = float(aout)
                table[str(model)] = entry
    _PRICING_CACHE = table
    return table


def get_pricing(model: str) -> Optional[Tuple[float, float]]:
    """Return (input_per_million_usd, output_per_million_usd) for model if known."""
    table = _load_pricing()
    entry = table.get(model)
    if not entry:
        return None
    return (entry["input_per_million"], entry["output_per_million"])


def get_pricing_details(model: str) -> Optional[Dict[str, float]]:
    """Return pricing details dict for model (may include cached input).

    Keys: input_per_million, output_per_million, cached_input_per_million (optional)
    """
    table = _load_pricing()
    entry = table.get(model)
    if not entry:
        return None
    return dict(entry)
