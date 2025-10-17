"""Tests for the admin log retrieval endpoints."""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from http import HTTPStatus
from typing import Tuple
from urllib.parse import quote

import pytest
from fastapi.testclient import TestClient

from bt_servant_engine.apps.api.app import create_app
from bt_servant_engine.apps.api.routes import admin_logs
from bt_servant_engine.bootstrap import build_default_service_container
from bt_servant_engine.core.config import config as app_config
import bt_servant_engine.core.logging as core_logging

ADMIN_TOKEN = "secret-token"
EXPECTED_FILE_COUNT = 2


def _auth_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {ADMIN_TOKEN}"}


@pytest.fixture(name="logs_client")
def _logs_client(monkeypatch, tmp_path) -> Tuple[TestClient, os.PathLike[str]]:
    """Provision a TestClient with admin auth and a temporary logs directory."""

    monkeypatch.setattr(app_config, "ENABLE_ADMIN_AUTH", True, raising=True)
    monkeypatch.setattr(app_config, "ADMIN_API_TOKEN", ADMIN_TOKEN, raising=True)

    monkeypatch.setattr(core_logging, "LOGS_DIR", tmp_path, raising=True)
    monkeypatch.setattr(core_logging, "LOG_FILE_PATH", tmp_path / "bt_servant.log", raising=True)
    monkeypatch.setattr(admin_logs, "LOGS_DIR", tmp_path, raising=True)

    client = TestClient(create_app(build_default_service_container()))
    return client, tmp_path


def test_list_logs_returns_sorted_entries(logs_client):
    """List endpoint returns sorted metadata and totals."""
    client, logs_dir = logs_client

    older_log = logs_dir / "older.log"
    older_log.write_text("older")
    older_ts = (datetime.now(timezone.utc) - timedelta(days=2)).timestamp()
    os.utime(older_log, (older_ts, older_ts))

    recent_log = logs_dir / "recent.log"
    recent_log.write_text("recent")

    resp = client.get("/admin/logs/files", headers=_auth_headers())
    assert resp.status_code == HTTPStatus.OK
    payload = resp.json()

    assert payload["total_files"] == EXPECTED_FILE_COUNT
    assert payload["total_size_bytes"] == sum(
        path.stat().st_size for path in (older_log, recent_log)
    )
    names = [entry["name"] for entry in payload["files"]]
    assert names == ["recent.log", "older.log"]
    assert payload["files"][0]["modified_at"].endswith("Z")


def test_log_endpoints_require_authentication(logs_client):
    """Missing credentials should yield 401."""
    client, logs_dir = logs_client
    (logs_dir / "sample.log").write_text("data")

    resp = client.get("/admin/logs/files")
    assert resp.status_code == HTTPStatus.UNAUTHORIZED


def test_download_log_file_streams_content(logs_client):
    """Downloading a log streams raw text with headers."""
    client, logs_dir = logs_client
    log_path = logs_dir / "example.log"
    log_path.write_text("line1\nline2\n")

    resp = client.get("/admin/logs/files/example.log", headers=_auth_headers())
    assert resp.status_code == HTTPStatus.OK
    assert resp.text == "line1\nline2\n"
    assert resp.headers["content-disposition"].endswith('example.log"')
    assert resp.headers["content-length"] == str(log_path.stat().st_size)


def test_download_rejects_path_traversal(logs_client):
    """Guard against path traversal attempts."""
    client, logs_dir = logs_client
    (logs_dir / "valid.log").write_text("content")

    encoded = quote("../secret.log", safe="")
    resp = client.get(f"/admin/logs/files/{encoded}", headers=_auth_headers())
    assert resp.status_code in {HTTPStatus.BAD_REQUEST, HTTPStatus.NOT_FOUND}


def test_recent_logs_filters_by_days_and_limit(logs_client):
    """Recent endpoint respects day window and limit."""
    client, logs_dir = logs_client

    old_log = logs_dir / "old.log"
    old_log.write_text("old")
    old_ts = (datetime.now(timezone.utc) - timedelta(days=10)).timestamp()
    os.utime(old_log, (old_ts, old_ts))

    new_log = logs_dir / "new.log"
    new_log.write_text("new")

    resp = client.get("/admin/logs/recent?days=7&limit=1", headers=_auth_headers())
    assert resp.status_code == HTTPStatus.OK
    payload = resp.json()

    assert payload["total_files"] == 1
    assert payload["files"][0]["name"] == "new.log"
    assert payload["total_size_bytes"] == new_log.stat().st_size
