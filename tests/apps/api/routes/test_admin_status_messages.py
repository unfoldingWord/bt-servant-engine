"""Tests for admin status message routes."""

from __future__ import annotations

import copy
import json
from http import HTTPStatus
from typing import Tuple

import pytest
from fastapi.testclient import TestClient

from bt_servant_engine.apps.api.app import create_app
from bt_servant_engine.bootstrap import build_default_service_container
from bt_servant_engine.core.config import config as app_config
from bt_servant_engine.services import status_messages

ADMIN_TOKEN = "secret-token"


def _auth_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {ADMIN_TOKEN}"}


@pytest.fixture(name="status_client")
def _status_client(monkeypatch: pytest.MonkeyPatch, tmp_path) -> Tuple[TestClient, str]:
    """Provision a TestClient with admin auth and a temporary status file."""
    monkeypatch.setattr(app_config, "ENABLE_ADMIN_AUTH", True, raising=True)
    monkeypatch.setattr(app_config, "ADMIN_API_TOKEN", ADMIN_TOKEN, raising=True)

    data = {
        "TEST_KEY": {"en": "English source", "am": "am text"},
        "OTHER_KEY": {"en": "Other English source", "am": "other am"},
    }
    path = tmp_path / "status_messages_data.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    store = status_messages.StatusMessageStore(
        status_messages=copy.deepcopy(data),
        dynamic_cache={("TEST_KEY", "am"): "cached am", ("OTHER_KEY", "am"): "cached other am"},
    )
    monkeypatch.setattr(status_messages, "_STATUS_MESSAGES_PATH", path, raising=True)
    monkeypatch.setattr(status_messages, "_STATUS_STORE", store, raising=True)

    client = TestClient(create_app(build_default_service_container()))
    return client, path


def test_delete_status_messages_by_language(status_client) -> None:
    """Deleting by language clears all keys, cache, and returns 204."""
    client, path = status_client

    # Pre-check ensure entries exist
    resp = client.get("/admin/status-messages/language/am", headers=_auth_headers())
    assert resp.status_code == HTTPStatus.OK
    assert resp.json()["translations"] == {"TEST_KEY": "am text", "OTHER_KEY": "other am"}

    resp = client.delete("/admin/status-messages/language/am", headers=_auth_headers())
    assert resp.status_code == HTTPStatus.NO_CONTENT

    body = json.loads(path.read_text(encoding="utf-8"))
    assert "am" not in body["TEST_KEY"]
    assert "am" not in body["OTHER_KEY"]
    assert ("TEST_KEY", "am") not in status_messages.get_dynamic_translation_cache()

    resp = client.get("/admin/status-messages/language/am", headers=_auth_headers())
    assert resp.status_code == HTTPStatus.OK
    assert resp.json()["translations"] == {}

    resp = client.delete("/admin/status-messages/language/am", headers=_auth_headers())
    assert resp.status_code == HTTPStatus.NOT_FOUND
