"""Runtime service container registry for dependency inversion.

This tiny module provides a single place where the application can register the
active :class:`ServiceContainer` instance so that service-layer modules can
resolve their dependencies without reaching directly into adapter modules.

The registry is process-wide; the FastAPI app sets it during startup, and tests
may override it with in-memory stubs as needed.
"""

from __future__ import annotations

from typing import Optional

from . import ServiceContainer

_services: Optional[ServiceContainer] = None


def set_services(container: ServiceContainer) -> None:
    """Register the active service container."""
    global _services  # noqa: PLW0603 - intentional module-level registry
    _services = container


def get_services() -> ServiceContainer:
    """Return the registered service container or raise if missing."""
    if _services is None:
        raise RuntimeError("Service container has not been configured.")
    return _services


def clear_services() -> None:
    """Reset the registry (used primarily in tests)."""
    global _services  # noqa: PLW0603
    _services = None


__all__ = ["set_services", "get_services", "clear_services"]
