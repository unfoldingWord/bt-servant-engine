"""Tests for FastAPI app factory."""
# pylint: disable=missing-function-docstring

import asyncio

from bt_servant_engine.apps.api.app import create_app, get_brain, lifespan, set_brain


def test_create_app_has_routes():
    app = create_app()
    paths = {
        path
        for route in app.router.routes
        if (path := getattr(route, "path", getattr(route, "path_format", "")))
    }
    assert "/alive" in paths
    assert any(path.startswith("/chroma") for path in paths)
    assert hasattr(app.state, "services")
    assert app.state.services.intent_router is not None
    assert app.state.services.chroma is not None
    assert app.state.services.user_state is not None
    assert app.state.services.messaging is not None


def test_lifespan_initializes_brain():
    app = create_app()

    async def _exercise() -> None:
        async with lifespan(app):
            assert get_brain() is not None

    asyncio.run(_exercise())
    set_brain(None)
