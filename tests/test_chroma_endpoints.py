"""Chroma endpoints test suite with FastAPI TestClient and monkeypatching."""
# pylint: disable=redefined-builtin
import chromadb
from fastapi.testclient import TestClient

import db.chroma_db as cdb
import bt_servant as api


class DummyEmbeddingFunction:
    """Simple callable stub returning fixed-size zero embeddings for inputs."""

    def __call__(self, input):  # type: ignore[override]
        return [[0.0, 0.0, 0.0] for _ in input]

    def ping(self) -> None:
        """No-op method to satisfy pylint public-methods threshold."""
        return None

    def is_legacy(self) -> bool:
        """Signal that this is a legacy-style embedder to keep Chroma happy."""
        return True

    @staticmethod
    def name() -> str:
        """Match OpenAI embedding function static interface."""
        return "dummy-embedding"


def make_tmp_client(tmp_path):
    """Create a chromadb PersistentClient rooted at a tmp path."""
    return chromadb.PersistentClient(path=str(tmp_path))


def test_create_and_delete_collection(monkeypatch, tmp_path):
    """Create collection, idempotency on duplicate, and delete semantics."""
    # Patch Chroma client and embedding function used by db helpers

    tmp_client = make_tmp_client(tmp_path)
    monkeypatch.setattr(cdb, "_aquifer_chroma_db", tmp_client)
    monkeypatch.setattr(cdb, "openai_ef", DummyEmbeddingFunction())

    # Patch startup to avoid brain initialization

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
    err = resp.json()["error"]
    assert ("Invalid" in err) or ("must be non-empty" in err)


def test_list_collections_and_delete_document(monkeypatch, tmp_path):
    """List collections and delete a previously added document."""
    # Patch Chroma client and embedding function used by db helpers

    tmp_client = chromadb.PersistentClient(path=str(tmp_path))
    monkeypatch.setattr(cdb, "_aquifer_chroma_db", tmp_client)
    monkeypatch.setattr(cdb, "openai_ef", DummyEmbeddingFunction())

    # Patch startup to avoid brain initialization

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
    """Admin auth enabled without credentials returns 401 JSON body."""
    # Ensure auth is enabled and no valid token is provided
    # api imported at module scope

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
    assert data.get("detail") in {
        "Missing credentials",
        "Admin token not configured",
        "Invalid credentials",
    }


def test_chroma_root_unauthorized_returns_401(monkeypatch):
    """Unauthorized request to /chroma returns 401 and WWW-Authenticate header."""
    # Unauthorized access to /chroma should yield 401 (not 404)
    # api imported at module scope

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
    assert data.get("detail") in {
        "Missing credentials",
        "Admin token not configured",
        "Invalid credentials",
    }


def test_count_documents_in_collection(monkeypatch, tmp_path):
    """Count documents in a collection via endpoint."""
    # Patch Chroma client and embedding function used by db helpers

    tmp_client = chromadb.PersistentClient(path=str(tmp_path))
    monkeypatch.setattr(cdb, "_aquifer_chroma_db", tmp_client)
    monkeypatch.setattr(cdb, "openai_ef", DummyEmbeddingFunction())

    # Patch startup to avoid brain initialization

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
    """List document IDs for a collection via endpoint."""
    # Patch Chroma client and embedding function used by db helpers

    tmp_client = chromadb.PersistentClient(path=str(tmp_path))
    monkeypatch.setattr(cdb, "_aquifer_chroma_db", tmp_client)
    monkeypatch.setattr(cdb, "openai_ef", DummyEmbeddingFunction())

    # Patch startup to avoid brain initialization

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
