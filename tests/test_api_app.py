"""Tests for FastAPI app factory."""
# pylint: disable=missing-function-docstring

import asyncio
from http import HTTPStatus

from fastapi.testclient import TestClient

from bt_servant_engine.apps.api.app import create_app, get_brain, lifespan, set_brain
from bt_servant_engine.services import runtime


def test_create_app_has_routes():
    app = create_app(runtime.get_services())
    paths = {
        path
        for route in app.router.routes
        if (path := getattr(route, "path", getattr(route, "path_format", "")))
    }
    assert "/alive" in paths
    assert any(path.startswith("/admin/chroma") for path in paths)
    assert hasattr(app.state, "services")
    assert app.state.services.intent_router is not None
    assert app.state.services.chroma is not None
    assert app.state.services.user_state is not None


def test_lifespan_initializes_brain():
    app = create_app(runtime.get_services())

    async def _exercise() -> None:
        async with lifespan(app):
            assert get_brain() is not None

    asyncio.run(_exercise())
    set_brain(None)


def test_correlation_id_middleware_roundtrips_header():
    client = TestClient(create_app(runtime.get_services()))

    resp = client.get("/alive")
    assert resp.status_code == HTTPStatus.OK
    request_id = resp.headers.get("X-Request-ID")
    assert request_id
    assert resp.headers.get("X-Correlation-ID") == request_id

    custom = "request-123"
    resp2 = client.get("/alive", headers={"X-Request-ID": custom})
    assert resp2.status_code == HTTPStatus.OK
    assert resp2.headers.get("X-Request-ID") == custom
    assert resp2.headers.get("X-Correlation-ID") == custom
