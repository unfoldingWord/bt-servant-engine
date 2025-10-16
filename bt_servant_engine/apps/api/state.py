"""Shared process-wide state for the API layer."""

from types import SimpleNamespace
from typing import Any

_STATE = SimpleNamespace(brain=None)


def get_brain() -> Any | None:
    """Return the cached brain instance, if initialized."""
    return _STATE.brain


def set_brain(value: Any | None) -> None:
    """Update the cached brain instance."""
    _STATE.brain = value


__all__ = ["get_brain", "set_brain"]
