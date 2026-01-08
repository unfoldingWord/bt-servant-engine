"""Core data transfer objects shared across layers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from pydantic import BaseModel


@dataclass(slots=True)
class RequestContext:
    """Metadata describing an inbound API request."""

    correlation_id: str
    path: str
    method: str
    user_agent: Optional[str] = None
    client_ip: Optional[str] = None


class PassageRef(BaseModel):
    """Normalized reference to a passage within a single canonical book."""

    book: str
    start_chapter: int | None = None
    start_verse: int | None = None
    end_chapter: int | None = None
    end_verse: int | None = None


class PassageSelection(BaseModel):
    """Structured selection consisting of one or more ranges for a book."""

    selections: List[PassageRef]


__all__ = [
    "RequestContext",
    "PassageRef",
    "PassageSelection",
]
