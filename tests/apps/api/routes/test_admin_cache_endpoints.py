"""Tests for admin cache management endpoints."""

# pylint: disable=missing-function-docstring

from __future__ import annotations

from http import HTTPStatus

from fastapi.testclient import TestClient

from bt_servant_engine.apps.api.app import create_app
from bt_servant_engine.apps.api.routes import admin_datastore as admin_datastore_router
from bt_servant_engine.bootstrap import build_default_service_container
from bt_servant_engine.core.config import config as app_config
from bt_servant_engine.services.admin import datastore as admin_datastore_service

MAX_CACHE_SAMPLE_LIMIT = admin_datastore_router.MAX_CACHE_SAMPLE_LIMIT
ADMIN_PREFIX = "/admin"


class _StubCache:
    def __init__(self) -> None:
        self.cleared = False
        self.sample_limit = 0

    def clear(self) -> None:
        self.cleared = True

    def stats(self) -> dict[str, object]:
        return {
            "name": "selection",
            "enabled": True,
            "ttl_seconds": 100,
            "max_entries": 10,
            "entry_count": 1,
            "bytes_used": 0,
            "oldest_entry_epoch": None,
            "newest_entry_epoch": None,
            "stats": {"hits": 1, "misses": 0, "stores": 1, "evictions": 0},
        }

    def detailed_stats(self, sample_limit: int = 10) -> dict[str, object]:
        self.sample_limit = sample_limit
        data = self.stats()
        data["samples"] = [
            {
                "key_repr": "dummy-key",
                "size_bytes": 42,
                "created_at": 1.0,
                "expires_at": 1000.0,
                "last_access": 10.0,
                "age_seconds": 9.0,
                "ttl_remaining": 991.0,
            }
        ]
        return data


class _StubCacheManager:
    def __init__(self) -> None:
        self.clear_all_called = False
        self.clear_called: list[str] = []
        self.prune_all_cutoff: float | None = None
        self.prune_cache_calls: list[tuple[str, float]] = []
        self.cache_obj = _StubCache()

    def clear_all(self) -> None:
        self.clear_all_called = True

    def clear_cache(self, name: str) -> None:
        if name != "selection":
            raise KeyError(name)
        self.clear_called.append(name)

    def stats(self) -> dict[str, object]:
        return {
            "enabled": True,
            "backend": "memory",
            "disk_root": "/tmp/cache",
            "disk_max_bytes": 512,
            "caches": {"selection": self.cache_obj.stats()},
        }

    def cache(self, name: str) -> _StubCache:
        if name != "selection":
            raise KeyError(name)
        return self.cache_obj

    def prune_all(self, cutoff: float) -> dict[str, int]:
        self.prune_all_cutoff = cutoff
        return {"selection": 1}

    def prune_cache(self, name: str, cutoff: float) -> int:
        if name != "selection":
            raise KeyError(name)
        self.prune_cache_calls.append((name, cutoff))
        return 2


def _make_client(monkeypatch) -> tuple[TestClient, _StubCacheManager]:
    stub = _StubCacheManager()
    monkeypatch.setattr(admin_datastore_service, "cache_manager", stub, raising=True)
    monkeypatch.setattr(app_config, "ENABLE_ADMIN_AUTH", False)
    client = TestClient(create_app(build_default_service_container()))
    return client, stub


def test_clear_all_caches_endpoint(monkeypatch):
    client, stub = _make_client(monkeypatch)
    resp = client.post(f"{ADMIN_PREFIX}/cache/clear")
    assert resp.status_code == HTTPStatus.OK
    assert resp.json()["status"] == "cleared"
    assert stub.clear_all_called


def test_clear_named_cache_endpoint(monkeypatch):
    client, stub = _make_client(monkeypatch)
    resp = client.post(f"{ADMIN_PREFIX}/cache/selection/clear")
    assert resp.status_code == HTTPStatus.OK
    assert resp.json()["cache"] == "selection"
    assert stub.clear_called == ["selection"]

    resp = client.post(f"{ADMIN_PREFIX}/cache/unknown/clear")
    assert resp.status_code == HTTPStatus.NOT_FOUND
    assert resp.json()["detail"]["error"] == "Cache 'unknown' not found"


def test_prune_all_caches_endpoint(monkeypatch):
    client, stub = _make_client(monkeypatch)
    resp = client.post(f"{ADMIN_PREFIX}/cache/clear", params={"older_than_days": 1})
    assert resp.status_code == HTTPStatus.OK
    payload = resp.json()
    assert payload["status"] == "pruned"
    assert "cutoff_epoch" in payload
    assert payload["removed"] == {"selection": 1}
    assert stub.prune_all_cutoff is not None


def test_prune_named_cache_endpoint(monkeypatch):
    client, stub = _make_client(monkeypatch)
    days = 2
    resp = client.post(f"{ADMIN_PREFIX}/cache/selection/clear", params={"older_than_days": days})
    assert resp.status_code == HTTPStatus.OK
    payload = resp.json()
    assert payload["status"] == "pruned"
    removed_expected = 2
    assert payload["removed"] == removed_expected
    assert stub.prune_cache_calls and stub.prune_cache_calls[0][1] > 0


def test_prune_invalid_params(monkeypatch):
    client, _ = _make_client(monkeypatch)
    resp = client.post(f"{ADMIN_PREFIX}/cache/clear", params={"older_than_days": -1})
    assert resp.status_code == HTTPStatus.BAD_REQUEST
    resp = client.post(f"{ADMIN_PREFIX}/cache/selection/clear", params={"older_than_days": 0})
    assert resp.status_code == HTTPStatus.BAD_REQUEST


def test_get_cache_stats_endpoint(monkeypatch):
    client, _ = _make_client(monkeypatch)
    resp = client.get(f"{ADMIN_PREFIX}/cache/stats")
    assert resp.status_code == HTTPStatus.OK
    data = resp.json()
    assert data["enabled"] is True
    assert "selection" in data["caches"]


def test_inspect_cache_endpoint(monkeypatch):
    client, stub = _make_client(monkeypatch)
    sample_limit = 5
    resp = client.get(f"{ADMIN_PREFIX}/cache/selection", params={"sample_limit": sample_limit})
    assert resp.status_code == HTTPStatus.OK
    data = resp.json()
    assert data["name"] == "selection"
    assert data["sample_limit"] == sample_limit
    assert stub.cache_obj.sample_limit == sample_limit
    assert data["samples"]

    bad_resp = client.get(f"{ADMIN_PREFIX}/cache/selection", params={"sample_limit": 0})
    assert bad_resp.status_code == HTTPStatus.BAD_REQUEST
    assert (
        bad_resp.json()["detail"]["error"]
        == f"sample_limit must be between 1 and {MAX_CACHE_SAMPLE_LIMIT}"
    )

    missing = client.get(f"{ADMIN_PREFIX}/cache/missing")
    assert missing.status_code == HTTPStatus.NOT_FOUND
    assert missing.json()["detail"]["error"] == "Cache 'missing' not found"
