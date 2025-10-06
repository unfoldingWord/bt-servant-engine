"""Shared process-wide state for the API layer."""

from typing import Any

_brain: Any | None = None


def get_brain() -> Any | None:
    """Return the cached brain instance, if initialized."""
    return _brain


def set_brain(value: Any | None) -> None:
    """Update the cached brain instance."""
    global _brain  # pylint: disable=global-statement
    _brain = value


__all__ = ["get_brain", "set_brain"]
