"""API factory entrypoint wiring default adapters to the FastAPI app."""

from __future__ import annotations

from bt_servant_engine.apps.api.app import create_app as _create_app
from bt_servant_engine.bootstrap import build_default_service_container


def create_app():  # noqa: D401 - FastAPI factory signature
    """Return a FastAPI app configured with the default service container."""

    return _create_app(build_default_service_container())


__all__ = ["create_app"]
