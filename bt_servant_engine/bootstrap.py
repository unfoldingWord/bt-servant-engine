"""Application bootstrap helpers for assembling the service container."""

from __future__ import annotations

from bt_servant_engine.adapters.chroma import ChromaAdapter
from bt_servant_engine.adapters.user_state import UserStateAdapter
from bt_servant_engine.services import ServiceContainer, build_default_services


def build_default_service_container() -> ServiceContainer:
    """Return the default service container wired to production adapters."""

    return build_default_services(
        chroma_port=ChromaAdapter(),
        user_state_port=UserStateAdapter(),
    )


__all__ = ["build_default_service_container"]
