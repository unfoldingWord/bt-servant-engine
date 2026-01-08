"""FastAPI application factory and shared API state."""

from __future__ import annotations

import asyncio
import concurrent.futures
from contextlib import asynccontextmanager

from fastapi import FastAPI

from bt_servant_engine.apps.api.middleware import CorrelationIdMiddleware
from bt_servant_engine.apps.api.routes import (
    admin_datastore,
    admin_logs,
    admin_status_messages,
    chat,
    health,
    users,
)
from bt_servant_engine.apps.api.user_locks import lock_cleanup_task
from bt_servant_engine.core.logging import get_logger
from bt_servant_engine.services.brain_orchestrator import create_brain
from bt_servant_engine.services import ServiceContainer
from bt_servant_engine.services import runtime
from .state import get_brain, set_brain

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize shared resources at startup and clean up on shutdown."""
    logger.info("Initializing bt servant engine...")
    logger.info("Loading brain...")
    loop = asyncio.get_event_loop()
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=64)
    loop.set_default_executor(executor)
    # Ensure runtime registry is populated even for non-HTTP contexts (e.g., CLI tests)
    services = getattr(app.state, "services", None)
    if isinstance(services, ServiceContainer):
        runtime.set_services(services)
    set_brain(create_brain())
    logger.info("brain loaded.")

    # Start background task to clean up stale user locks
    cleanup_task = asyncio.create_task(lock_cleanup_task())
    logger.info("User lock cleanup task started.")

    try:
        yield
    finally:
        cleanup_task.cancel()
        try:
            await cleanup_task
        except asyncio.CancelledError:
            pass
        executor.shutdown(wait=False)


def create_app(services: ServiceContainer | None = None) -> FastAPI:
    """Build the FastAPI application with configured routers."""
    if services is None:
        raise RuntimeError("Service container must be provided when creating the app.")
    app = FastAPI(lifespan=lifespan)
    service_container = services
    app.state.services = service_container
    runtime.set_services(service_container)
    app.add_middleware(CorrelationIdMiddleware)

    app.include_router(health.router)
    app.include_router(admin_logs.router)
    app.include_router(admin_status_messages.router)
    app.include_router(admin_datastore.router)
    app.include_router(chat.router)
    app.include_router(users.router)
    return app


__all__ = ["create_app", "get_brain", "set_brain", "lifespan"]
