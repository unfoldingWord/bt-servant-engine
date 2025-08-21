import os
import sys
from pathlib import Path

import chromadb
from fastapi.testclient import TestClient

# Ensure repository root is on sys.path for local package imports
sys.path.append(str(Path(__file__).resolve().parents[1]))


# Ensure required env vars exist for config import in app
os.environ.setdefault("OPENAI_API_KEY", "test")
os.environ.setdefault("GROQ_API_KEY", "test")
os.environ.setdefault("META_VERIFY_TOKEN", "test")
os.environ.setdefault("META_WHATSAPP_TOKEN", "test")
os.environ.setdefault("META_PHONE_NUMBER_ID", "test")
os.environ.setdefault("META_APP_SECRET", "test")
os.environ.setdefault("FACEBOOK_USER_AGENT", "test")
os.environ.setdefault("BASE_URL", "http://example.com")


class DummyEmbeddingFunction:
    def __call__(self, input):  # type: ignore[override]
        return [[0.0, 0.0, 0.0] for _ in input]


def make_tmp_client(tmp_path: Path) -> chromadb.PersistentClient:
    return chromadb.PersistentClient(path=str(tmp_path))


def test_create_and_delete_collection(monkeypatch, tmp_path):
    # Patch Chroma client and embedding function used by db helpers
    import db.chroma_db as cdb

    tmp_client = make_tmp_client(tmp_path)
    monkeypatch.setattr(cdb, "_aquifer_chroma_db", tmp_client)
    monkeypatch.setattr(cdb, "openai_ef", DummyEmbeddingFunction())

    # Import app and patch startup to avoid brain initialization
    import bt_servant as api

    def noop_init():  # no-op startup
        return None

    monkeypatch.setattr(api, "init", noop_init)

    client = TestClient(api.app)

    # Create a collection
    resp = client.post("/chroma/collections", json={"name": "testcol"})
    assert resp.status_code == 201, resp.text
    assert resp.json()["status"] == "created"

    # Creating the same collection again should yield 409
    resp = client.post("/chroma/collections", json={"name": "testcol"})
    assert resp.status_code == 409
    assert resp.json()["error"] == "Collection already exists"

    # Delete the collection
    resp = client.delete("/chroma/collections/testcol")
    assert resp.status_code == 204

    # Deleting again should yield 404
    resp = client.delete("/chroma/collections/testcol")
    assert resp.status_code == 404
    assert resp.json()["error"] == "Collection not found"

    # Invalid name cases
    resp = client.post("/chroma/collections", json={"name": "   "})
    assert resp.status_code == 400
    assert "Invalid" in resp.json()["error"] or "must be non-empty" in resp.json()["error"]
