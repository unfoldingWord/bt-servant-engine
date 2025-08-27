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


def test_list_collections_and_delete_document(monkeypatch, tmp_path):
    # Patch Chroma client and embedding function used by db helpers
    import db.chroma_db as cdb

    tmp_client = chromadb.PersistentClient(path=str(tmp_path))
    monkeypatch.setattr(cdb, "_aquifer_chroma_db", tmp_client)
    monkeypatch.setattr(cdb, "openai_ef", DummyEmbeddingFunction())

    # Import app and patch startup to avoid brain initialization
    import bt_servant as api

    def noop_init():
        return None

    monkeypatch.setattr(api, "init", noop_init)
    client = TestClient(api.app)

    # Initially, list is empty
    resp = client.get("/chroma/collections")
    assert resp.status_code == 200
    assert resp.json()["collections"] == []

    # Create a collection
    resp = client.post("/chroma/collections", json={"name": "col1"})
    assert resp.status_code == 201

    # List should include col1
    resp = client.get("/chroma/collections")
    assert resp.status_code == 200
    assert "col1" in resp.json()["collections"]

    # Add a document via existing endpoint
    doc_payload = {
        "document_id": "42",
        "collection": "col1",
        "name": "Test Doc",
        "text": "Hello world",
        "metadata": {"k": "v"},
    }
    resp = client.post("/add-document", json=doc_payload)
    assert resp.status_code == 200

    # Delete the document
    resp = client.delete("/chroma/collections/col1/documents/42")
    assert resp.status_code == 204

    # Deleting again should yield 404
    resp = client.delete("/chroma/collections/col1/documents/42")
    assert resp.status_code == 404
    assert resp.json()["error"] == "Document not found"


def test_admin_auth_401_json_body(monkeypatch):
    # Ensure auth is enabled and no valid token is provided
    import bt_servant as api

    def noop_init():
        return None

    monkeypatch.setattr(api, "init", noop_init)
    monkeypatch.setattr(api.config, "ENABLE_ADMIN_AUTH", True)
    monkeypatch.setattr(api.config, "ADMIN_API_TOKEN", "secret")

    client = TestClient(api.app)
    # No Authorization headers provided
    resp = client.get("/chroma/collections")
    assert resp.status_code == 401
    # JSON body present with detail
    assert resp.headers.get("WWW-Authenticate") == "Bearer"
    data = resp.json()
    assert data.get("detail") in {"Missing credentials", "Admin token not configured", "Invalid credentials"}


def test_chroma_root_unauthorized_returns_401(monkeypatch):
    # Unauthorized access to /chroma should yield 401 (not 404)
    import bt_servant as api

    def noop_init():
        return None

    monkeypatch.setattr(api, "init", noop_init)
    monkeypatch.setattr(api.config, "ENABLE_ADMIN_AUTH", True)
    monkeypatch.setattr(api.config, "ADMIN_API_TOKEN", "secret")

    client = TestClient(api.app)
    resp = client.get("/chroma")
    assert resp.status_code == 401
    assert resp.headers.get("WWW-Authenticate") == "Bearer"
    data = resp.json()
    assert data.get("detail") in {"Missing credentials", "Admin token not configured", "Invalid credentials"}


def test_count_documents_in_collection(monkeypatch, tmp_path):
    # Patch Chroma client and embedding function used by db helpers
    import db.chroma_db as cdb

    tmp_client = chromadb.PersistentClient(path=str(tmp_path))
    monkeypatch.setattr(cdb, "_aquifer_chroma_db", tmp_client)
    monkeypatch.setattr(cdb, "openai_ef", DummyEmbeddingFunction())

    # Import app and patch startup to avoid brain initialization
    import bt_servant as api

    def noop_init():
        return None

    monkeypatch.setattr(api, "init", noop_init)
    client = TestClient(api.app)

    # Missing collection -> 404
    resp = client.get("/chroma/collections/missing/count")
    assert resp.status_code == 404
    assert resp.json()["error"] == "Collection not found"

    # Create a collection and add documents directly via client
    cdb.create_chroma_collection("countcol")
    collection = cdb.get_or_create_chroma_collection("countcol")
    collection.upsert(
        ids=["1", "2", "3"],
        documents=["a", "b", "c"],
        metadatas=[{"m": "1"}, {"m": "2"}, {"m": "3"}],
    )

    # Count should be 3
    resp = client.get("/chroma/collections/countcol/count")
    assert resp.status_code == 200
    body = resp.json()
    assert body["collection"] == "countcol"
    assert body["count"] == 3


def test_list_document_ids_endpoint(monkeypatch, tmp_path):
    # Patch Chroma client and embedding function used by db helpers
    import db.chroma_db as cdb

    tmp_client = chromadb.PersistentClient(path=str(tmp_path))
    monkeypatch.setattr(cdb, "_aquifer_chroma_db", tmp_client)
    monkeypatch.setattr(cdb, "openai_ef", DummyEmbeddingFunction())

    # Import app and patch startup to avoid brain initialization
    import bt_servant as api

    def noop_init():
        return None

    monkeypatch.setattr(api, "init", noop_init)
    client = TestClient(api.app)

    # Missing collection -> 404
    resp = client.get("/chroma/collection/missing/ids")
    assert resp.status_code == 404
    assert resp.json()["error"] == "Collection not found"

    # Create a collection and add documents directly via client
    cdb.create_chroma_collection("idscol")
    collection = cdb.get_or_create_chroma_collection("idscol")
    collection.upsert(
        ids=["tn_ACT_vcsw", "2", "3"],
        documents=["a", "b", "c"],
        metadatas=[{"m": "a"}, {"m": "b"}, {"m": "c"}],
    )

    # Fetch ids via endpoint
    resp = client.get("/chroma/collection/idscol/ids")
    assert resp.status_code == 200
    body = resp.json()
    assert body["collection"] == "idscol"
    assert body["count"] == 3
    assert sorted(body["ids"]) == sorted(["tn_ACT_vcsw", "2", "3"])
