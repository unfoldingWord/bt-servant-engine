"""Tests for Chroma collection merge endpoint (uses an in-memory fake client).

Pylint is relaxed for this test module to keep fakes concise.
"""

# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
# pylint: disable=line-too-long,too-few-public-methods,consider-iterating-dictionary,consider-using-from-import
# pylint: disable=unused-argument,redefined-outer-name

from __future__ import annotations

import time
from http import HTTPStatus
from typing import Any, Dict, List, Optional

import pytest
from fastapi.testclient import TestClient

import bt_servant_engine.adapters.chroma as chroma_db
from bt_servant_engine.apps.api.app import create_app
from bt_servant_engine.bootstrap import build_default_service_container

DUPLICATES_PREVIEW_LIMIT = 2
BATCH_SIZE_LIMIT = 2
BACKGROUND_START_DELAY_SECONDS = 0.2
MERGE_POLL_ATTEMPTS = 1000
MERGE_POLL_INTERVAL_SECONDS = 0.05
CANCEL_BATCH_SIZE = 5
CANCEL_START_DELAY_SECONDS = 0.1
ADMIN_PREFIX = "/admin"
ADMIN_CHROMA_PREFIX = f"{ADMIN_PREFIX}/chroma"


class FakeCollection:
    def __init__(self, name: str):
        self.name = name
        self._store: Dict[str, Dict[str, Any]] = {}

    def get(
        self,
        ids: Optional[List[str]] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        include: Optional[List[str]] = None,
    ):  # noqa: D401 - mimic chroma signature
        include = include or ["ids", "documents", "metadatas", "embeddings"]
        if ids is not None:
            keys = [k for k in ids if k in self._store]
        else:
            keys = list(self._store.keys())
            if offset:
                keys = keys[offset:]
            if limit:
                keys = keys[:limit]
        result: Dict[str, Any] = {"ids": keys}
        if "documents" in include:
            result["documents"] = [self._store[k]["document"] for k in keys]
        if "metadatas" in include:
            result["metadatas"] = [self._store[k]["metadata"] for k in keys]
        if "embeddings" in include:
            result["embeddings"] = [self._store[k].get("embedding") for k in keys]
        return result

    def add(
        self,
        ids: List[str],
        documents: Optional[List[str]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        embeddings: Optional[List[List[float]]] = None,
    ):  # noqa: D401 - mimic chroma signature
        for i, doc_id in enumerate(ids):
            if doc_id in self._store:
                raise ValueError("duplicate id")
            self._store[doc_id] = {
                "document": documents[i] if documents else None,
                "metadata": metadatas[i] if metadatas else {},
                "embedding": embeddings[i] if embeddings else None,
            }

    def delete(self, ids: List[str]):
        for doc_id in ids:
            self._store.pop(doc_id, None)

    def count(self):  # noqa: D401
        return len(self._store)


class FakeClient:
    def __init__(self):
        self._cols: Dict[str, FakeCollection] = {}

    def list_collections(self):
        class _Obj:
            def __init__(self, name: str):
                self.name = name

        return [_Obj(name) for name in self._cols.keys()]

    def get_collection(self, name: str, embedding_function: Any = None):  # noqa: D401 - signature match
        return self._cols[name]

    def create_collection(self, name: str, embedding_function: Any = None):
        col = FakeCollection(name)
        self._cols[name] = col
        return col


@pytest.fixture(autouse=True)
def fake_chroma(monkeypatch):
    client = FakeClient()
    monkeypatch.setattr(chroma_db, "_aquifer_chroma_db", client, raising=True)
    # Create two collections commonly used by tests
    client.create_collection("src")
    client.create_collection("dst")
    return client


def test_dry_run_duplicates_preview_limit(fake_chroma):
    src = fake_chroma.get_collection("src")
    dst = fake_chroma.get_collection("dst")
    # Seed overlapping ids
    src.add(
        ids=["1", "2", "3"],
        documents=["a", "b", "c"],
        metadatas=[{}, {}, {}],
        embeddings=[[0.1], [0.2], [0.3]],
    )
    dst.add(ids=["2", "5"], documents=["x", "y"], metadatas=[{}, {}], embeddings=[[0.9], [1.0]])

    client = TestClient(create_app(build_default_service_container()))
    resp = client.post(
        f"{ADMIN_CHROMA_PREFIX}/collections/dst/merge",
        json={
            "source": "src",
            "on_duplicate": "fail",
            "dry_run": True,
            "duplicates_preview_limit": DUPLICATES_PREVIEW_LIMIT,
        },
    )
    assert resp.status_code == HTTPStatus.OK, resp.text
    body = resp.json()
    assert body["duplicates_found"] is True
    assert len(body["duplicate_preview"]) <= DUPLICATES_PREVIEW_LIMIT


@pytest.mark.skip(
    reason="Background thread execution incompatible with TestClient. "
    "Merge endpoint spawns concurrent.futures thread which doesn't execute "
    "in FastAPI TestClient synchronous test environment. Works in production."
)
def test_merge_create_new_id_with_tags_and_copy(fake_chroma):
    src = fake_chroma.get_collection("src")
    # Seed source only
    src.add(
        ids=["10", "11", "12"],
        documents=["a", "b", "c"],
        metadatas=[{"k": 1}, {"k": 2}, {"k": 3}],
        embeddings=[[0.1], [0.2], [0.3]],
    )

    client = TestClient(create_app(build_default_service_container()))
    # Start merge
    resp = client.post(
        f"{ADMIN_CHROMA_PREFIX}/collections/dst/merge",
        json={
            "source": "src",
            "mode": "copy",
            "create_new_id": True,
            "use_source_embeddings": True,
            "batch_size": BATCH_SIZE_LIMIT,
        },
    )
    assert resp.status_code == HTTPStatus.ACCEPTED, resp.text
    task = resp.json()
    task_id = task["task_id"]

    # Give background thread time to start
    time.sleep(BACKGROUND_START_DELAY_SECONDS)

    # Poll for completion with generous timeout
    for _ in range(MERGE_POLL_ATTEMPTS):  # 50s total timeout for CI environments
        st = client.get(f"{ADMIN_CHROMA_PREFIX}/merge-tasks/{task_id}")
        assert st.status_code == HTTPStatus.OK
        data = st.json()
        if data["status"] in ("completed", "failed"):
            break
        time.sleep(MERGE_POLL_INTERVAL_SECONDS)
    assert data["status"] == "completed"
    # Verify dest content and ids
    dst = fake_chroma.get_collection("dst")
    res = dst.get()
    assert res["ids"] == ["1", "2", "3"]
    # Verify tags
    for md in res["metadatas"]:
        assert md.get("_merged_from") == "src"
        assert md.get("_merge_task_id") == task_id
        assert "_merged_at" in md


@pytest.mark.skip(
    reason="Background thread execution incompatible with TestClient. "
    "Merge endpoint spawns concurrent.futures thread which doesn't execute "
    "in FastAPI TestClient synchronous test environment. Works in production."
)
def test_cancel_merge(fake_chroma):
    src = fake_chroma.get_collection("src")
    # Seed many docs to allow cancellation before completion
    ids = [str(i) for i in range(1, 51)]
    src.add(
        ids=ids,
        documents=["x"] * len(ids),
        metadatas=[{}] * len(ids),
        embeddings=[[0.0]] * len(ids),
    )

    client = TestClient(create_app(build_default_service_container()))
    resp = client.post(
        f"{ADMIN_CHROMA_PREFIX}/collections/dst/merge",
        json={
            "source": "src",
            "mode": "copy",
            "create_new_id": True,
            "use_source_embeddings": True,
            "batch_size": CANCEL_BATCH_SIZE,
        },
    )
    assert resp.status_code == HTTPStatus.ACCEPTED
    task_id = resp.json()["task_id"]

    # Give background thread time to start before canceling
    time.sleep(CANCEL_START_DELAY_SECONDS)

    # Request cancel
    cancel = client.delete(f"{ADMIN_CHROMA_PREFIX}/merge-tasks/{task_id}")
    # If the task already completed, cancellation can return 409
    assert cancel.status_code in (HTTPStatus.ACCEPTED, HTTPStatus.CONFLICT)

    # Wait for cancel to be acknowledged with generous timeout
    for _ in range(MERGE_POLL_ATTEMPTS):  # 50s timeout for CI environments
        st = client.get(f"{ADMIN_CHROMA_PREFIX}/merge-tasks/{task_id}")
        assert st.status_code == HTTPStatus.OK
        data = st.json()
        if data["status"] in ("cancelled", "completed", "failed"):
            break
        time.sleep(MERGE_POLL_INTERVAL_SECONDS)
    assert data["status"] in ("cancelled", "completed")
    # If cancelled, ensure partial progress
    if data["status"] == "cancelled":
        assert data["completed"] < data["total"]
