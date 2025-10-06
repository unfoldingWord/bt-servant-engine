"""Unit tests for API dependency helpers."""
# pylint: disable=missing-function-docstring

import asyncio
import pytest
from fastapi import HTTPException

from config import config as app_config
from bt_servant_engine.apps.api import dependencies


def test_require_admin_token_disabled(monkeypatch):
    monkeypatch.setattr(app_config, "ENABLE_ADMIN_AUTH", False, raising=True)
    asyncio.run(dependencies.require_admin_token())


def test_require_admin_token_valid(monkeypatch):
    monkeypatch.setattr(app_config, "ENABLE_ADMIN_AUTH", True, raising=True)
    monkeypatch.setattr(app_config, "ADMIN_API_TOKEN", "secret", raising=True)
    asyncio.run(
        dependencies.require_admin_token(
            authorization="Bearer secret",
        )
    )


def test_require_admin_token_invalid(monkeypatch):
    monkeypatch.setattr(app_config, "ENABLE_ADMIN_AUTH", True, raising=True)
    monkeypatch.setattr(app_config, "ADMIN_API_TOKEN", "secret", raising=True)
    with pytest.raises(HTTPException):
        asyncio.run(dependencies.require_admin_token(authorization="Bearer nope"))


def test_require_healthcheck_token_valid(monkeypatch):
    monkeypatch.setattr(app_config, "ENABLE_ADMIN_AUTH", True, raising=True)
    monkeypatch.setattr(app_config, "HEALTHCHECK_API_TOKEN", "health", raising=True)
    asyncio.run(
        dependencies.require_healthcheck_token(
            x_admin_token="health",
        )
    )
