"""FastAPI application factory and shared API state."""

from __future__ import annotations

import asyncio
import concurrent.futures
from contextlib import asynccontextmanager
from fastapi import FastAPI

from brain import create_brain
from logger import get_logger
from bt_servant_engine.services import build_default_services

from .state import get_brain, set_brain

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Initialize shared resources at startup and clean up on shutdown."""
    logger.info("Initializing bt servant engine...")
    logger.info("Loading brain...")
    loop = asyncio.get_event_loop()
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=64)
    loop.set_default_executor(executor)
    set_brain(create_brain())
    logger.info("brain loaded.")
    try:
        yield
    finally:
        executor.shutdown(wait=False)


def create_app() -> FastAPI:
    """Build the FastAPI application with configured routers."""
    app = FastAPI(lifespan=lifespan)
    app.state.services = build_default_services()

    # Import lazily to avoid potential circular imports when routers grow.
    from .routes import admin, health, webhooks  # pylint: disable=import-outside-toplevel

    app.include_router(health.router)
    app.include_router(admin.router)
    app.include_router(webhooks.router)
    return app


__all__ = ["create_app", "get_brain", "set_brain", "lifespan"]
