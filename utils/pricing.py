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
    if raw:
        try:
            data = json.loads(raw)
            if isinstance(data, dict):
                for model, cfg in data.items():
                    if not isinstance(cfg, dict):
                        continue
                    inp = cfg.get("input_per_million")
                    out = cfg.get("output_per_million")
                    if isinstance(inp, (int, float)) and isinstance(out, (int, float)):
                        table[str(model)] = {
                            "input_per_million": float(inp),
                            "output_per_million": float(out),
                        }
        except (json.JSONDecodeError, TypeError, ValueError):
            # Ignore invalid env and leave table empty (no pricing)
            table = {}
    _PRICING_CACHE = table
    return table


def get_pricing(model: str) -> Optional[Tuple[float, float]]:
    """Return (input_per_million_usd, output_per_million_usd) for model if known."""
    table = _load_pricing()
    entry = table.get(model)
    if not entry:
        return None
    return (entry["input_per_million"], entry["output_per_million"])
