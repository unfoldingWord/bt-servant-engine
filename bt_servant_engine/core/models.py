"""Core data transfer objects shared across layers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(slots=True)
class RequestContext:
    """Metadata describing an inbound API request."""

    correlation_id: str
    path: str
    method: str
    user_agent: Optional[str] = None


__all__ = ["RequestContext"]
