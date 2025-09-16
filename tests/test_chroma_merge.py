"""Tests for Chroma collection merge endpoint (uses an in-memory fake client).

Pylint is relaxed for this test module to keep fakes concise.
"""

# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
# pylint: disable=line-too-long,too-few-public-methods,consider-iterating-dictionary,consider-using-from-import
# pylint: disable=unused-argument,redefined-outer-name

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import pytest
from fastapi.testclient import TestClient

import db.chroma_db as chroma_db
from bt_servant import app


class FakeCollection:
    def __init__(self, name: str):
        self.name = name
        self._store: Dict[str, Dict[str, Any]] = {}

    def get(self, ids: Optional[List[str]] = None, limit: Optional[int] = None, offset: Optional[int] = None, include: Optional[List[str]] = None):  # noqa: D401 - mimic chroma signature
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

    def add(self, ids: List[str], documents: Optional[List[str]] = None, metadatas: Optional[List[Dict[str, Any]]] = None, embeddings: Optional[List[List[float]]] = None):  # noqa: D401 - mimic chroma signature
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
    src.add(ids=["1", "2", "3"], documents=["a", "b", "c"], metadatas=[{}, {}, {}], embeddings=[[0.1], [0.2], [0.3]])
    dst.add(ids=["2", "5"], documents=["x", "y"], metadatas=[{}, {}], embeddings=[[0.9], [1.0]])

    client = TestClient(app)
    resp = client.post(
        "/chroma/collections/dst/merge",
        json={
            "source": "src",
            "on_duplicate": "fail",
            "dry_run": True,
            "duplicates_preview_limit": 2,
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["duplicates_found"] is True
    assert len(body["duplicate_preview"]) <= 2


def test_merge_create_new_id_with_tags_and_copy(fake_chroma):
    src = fake_chroma.get_collection("src")
    # Seed source only
    src.add(ids=["10", "11", "12"], documents=["a", "b", "c"], metadatas=[{"k": 1}, {"k": 2}, {"k": 3}], embeddings=[[0.1], [0.2], [0.3]])

    client = TestClient(app)
    # Start merge
    resp = client.post(
        "/chroma/collections/dst/merge",
        json={
            "source": "src",
            "mode": "copy",
            "create_new_id": True,
            "use_source_embeddings": True,
            "batch_size": 2,
        },
    )
    assert resp.status_code == 202, resp.text
    task = resp.json()
    task_id = task["task_id"]

    # Poll for completion
    for _ in range(250):
        st = client.get(f"/chroma/merge-tasks/{task_id}")
        assert st.status_code == 200
        data = st.json()
        if data["status"] in ("completed", "failed"):
            break
        time.sleep(0.02)
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


def test_cancel_merge(fake_chroma):
    src = fake_chroma.get_collection("src")
    # Seed many docs to allow cancellation before completion
    ids = [str(i) for i in range(1, 51)]
    src.add(ids=ids, documents=["x"] * len(ids), metadatas=[{}] * len(ids), embeddings=[[0.0]] * len(ids))

    client = TestClient(app)
    resp = client.post(
        "/chroma/collections/dst/merge",
        json={
            "source": "src",
            "mode": "copy",
            "create_new_id": True,
            "use_source_embeddings": True,
            "batch_size": 5,
        },
    )
    assert resp.status_code == 202
    task_id = resp.json()["task_id"]

    # Immediately request cancel
    cancel = client.delete(f"/chroma/merge-tasks/{task_id}")
    # If the task already completed, cancellation can return 409
    assert cancel.status_code in (202, 409)

    # Wait for cancel to be acknowledged
    for _ in range(50):
        st = client.get(f"/chroma/merge-tasks/{task_id}")
        assert st.status_code == 200
        data = st.json()
        if data["status"] in ("cancelled", "completed", "failed"):
            break
        time.sleep(0.02)
    assert data["status"] in ("cancelled", "completed")
    # If cancelled, ensure partial progress
    if data["status"] == "cancelled":
        assert data["completed"] < data["total"]
