"""Chroma endpoints test suite with FastAPI TestClient and monkeypatching."""

# pylint: disable=redefined-builtin
from http import HTTPStatus
from unittest.mock import patch

import chromadb
import pytest
from fastapi.testclient import TestClient

from bt_servant_engine.apps.api.app import create_app
from bt_servant_engine.bootstrap import build_default_service_container
import bt_servant_engine.adapters.chroma as cdb
from bt_servant_engine.core.config import config as app_config


@pytest.fixture
def disable_auth():
    """Disable authentication for tests that don't test authentication."""
    with patch("bt_servant_engine.apps.api.dependencies.config") as mock_config:
        mock_config.ENABLE_ADMIN_AUTH = False
        yield

EXPECTED_COLLECTION_DOCUMENT_COUNT = 3
ADMIN_PREFIX = "/admin"
ADMIN_CHROMA_PREFIX = f"{ADMIN_PREFIX}/chroma"


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


def test_create_and_delete_collection(monkeypatch, tmp_path, disable_auth):
    """Create collection, idempotency on duplicate, and delete semantics."""
    # Patch Chroma client and embedding function used by db helpers

    tmp_client = make_tmp_client(tmp_path)
    monkeypatch.setattr(cdb, "_aquifer_chroma_db", tmp_client)
    monkeypatch.setattr(cdb, "openai_ef", DummyEmbeddingFunction())

    # Patch startup to avoid brain initialization
    client = TestClient(create_app(build_default_service_container()))

    # Create a collection
    resp = client.post(f"{ADMIN_CHROMA_PREFIX}/collections", json={"name": "testcol"})
    assert resp.status_code == HTTPStatus.CREATED, resp.text
    assert resp.json()["status"] == "created"

    # Creating the same collection again should yield 409
    resp = client.post(f"{ADMIN_CHROMA_PREFIX}/collections", json={"name": "testcol"})
    assert resp.status_code == HTTPStatus.CONFLICT
    assert resp.json()["error"] == "Collection already exists"

    # Delete the collection
    resp = client.delete(f"{ADMIN_CHROMA_PREFIX}/collections/testcol")
    assert resp.status_code == HTTPStatus.NO_CONTENT

    # Deleting again should yield 404
    resp = client.delete(f"{ADMIN_CHROMA_PREFIX}/collections/testcol")
    assert resp.status_code == HTTPStatus.NOT_FOUND
    assert resp.json()["error"] == "Collection not found"

    # Invalid name cases
    resp = client.post(f"{ADMIN_CHROMA_PREFIX}/collections", json={"name": "   "})
    assert resp.status_code == HTTPStatus.BAD_REQUEST
    err = resp.json()["error"]
    assert ("Invalid" in err) or ("must be non-empty" in err)


def test_list_collections_and_delete_document(monkeypatch, tmp_path, disable_auth):
    """List collections and delete a previously added document."""
    # Patch Chroma client and embedding function used by db helpers

    tmp_client = chromadb.PersistentClient(path=str(tmp_path))
    monkeypatch.setattr(cdb, "_aquifer_chroma_db", tmp_client)
    monkeypatch.setattr(cdb, "openai_ef", DummyEmbeddingFunction())

    # Patch startup to avoid brain initialization
    client = TestClient(create_app(build_default_service_container()))

    # Initially, list is empty
    resp = client.get(f"{ADMIN_CHROMA_PREFIX}/collections")
    assert resp.status_code == HTTPStatus.OK
    assert resp.json()["collections"] == []

    # Create a collection
    resp = client.post(f"{ADMIN_CHROMA_PREFIX}/collections", json={"name": "col1"})
    assert resp.status_code == HTTPStatus.CREATED

    # List should include col1
    resp = client.get(f"{ADMIN_CHROMA_PREFIX}/collections")
    assert resp.status_code == HTTPStatus.OK
    assert "col1" in resp.json()["collections"]

    # Add a document via existing endpoint
    doc_payload = {
        "document_id": "42",
        "collection": "col1",
        "name": "Test Doc",
        "text": "Hello world",
        "metadata": {"k": "v"},
    }
    resp = client.post(f"{ADMIN_CHROMA_PREFIX}/add-document", json=doc_payload)
    assert resp.status_code == HTTPStatus.OK

    # Delete the document
    resp = client.delete(f"{ADMIN_CHROMA_PREFIX}/collections/col1/documents/42")
    assert resp.status_code == HTTPStatus.NO_CONTENT

    # Deleting again should yield 404
    resp = client.delete(f"{ADMIN_CHROMA_PREFIX}/collections/col1/documents/42")
    assert resp.status_code == HTTPStatus.NOT_FOUND
    assert resp.json()["error"] == "Document not found"


def test_admin_auth_401_json_body(monkeypatch):
    """Admin auth enabled without credentials returns 401 JSON body."""
    # Ensure auth is enabled and no valid token is provided
    # api imported at module scope

    monkeypatch.setattr(app_config, "ENABLE_ADMIN_AUTH", True)
    monkeypatch.setattr(app_config, "ADMIN_API_TOKEN", "secret")

    client = TestClient(create_app(build_default_service_container()))
    # No Authorization headers provided
    resp = client.get(f"{ADMIN_CHROMA_PREFIX}/collections")
    assert resp.status_code == HTTPStatus.UNAUTHORIZED
    # JSON body present with detail
    assert resp.headers.get("WWW-Authenticate") == "Bearer"
    data = resp.json()
    assert data.get("detail") in {
        "Missing credentials",
        "Admin token not configured",
        "Invalid credentials",
    }


def test_chroma_root_unauthorized_returns_401(monkeypatch):
    """Unauthorized request to /admin/chroma returns 401 and WWW-Authenticate header."""
    # Unauthorized access to /admin/chroma should yield 401 (not 404)
    # api imported at module scope

    monkeypatch.setattr(app_config, "ENABLE_ADMIN_AUTH", True)
    monkeypatch.setattr(app_config, "ADMIN_API_TOKEN", "secret")

    client = TestClient(create_app(build_default_service_container()))
    resp = client.get(ADMIN_CHROMA_PREFIX)
    assert resp.status_code == HTTPStatus.UNAUTHORIZED
    assert resp.headers.get("WWW-Authenticate") == "Bearer"
    data = resp.json()
    assert data.get("detail") in {
        "Missing credentials",
        "Admin token not configured",
        "Invalid credentials",
    }


def test_count_documents_in_collection(monkeypatch, tmp_path, disable_auth):
    """Count documents in a collection via endpoint."""
    # Patch Chroma client and embedding function used by db helpers

    tmp_client = chromadb.PersistentClient(path=str(tmp_path))
    monkeypatch.setattr(cdb, "_aquifer_chroma_db", tmp_client)
    monkeypatch.setattr(cdb, "openai_ef", DummyEmbeddingFunction())

    # Patch startup to avoid brain initialization
    client = TestClient(create_app(build_default_service_container()))

    # Missing collection -> 404
    resp = client.get(f"{ADMIN_CHROMA_PREFIX}/collections/missing/count")
    assert resp.status_code == HTTPStatus.NOT_FOUND
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
    resp = client.get(f"{ADMIN_CHROMA_PREFIX}/collections/countcol/count")
    assert resp.status_code == HTTPStatus.OK
    body = resp.json()
    assert body["collection"] == "countcol"
    assert body["count"] == EXPECTED_COLLECTION_DOCUMENT_COUNT


def test_list_document_ids_endpoint(monkeypatch, tmp_path, disable_auth):
    """List document IDs for a collection via endpoint."""
    # Patch Chroma client and embedding function used by db helpers

    tmp_client = chromadb.PersistentClient(path=str(tmp_path))
    monkeypatch.setattr(cdb, "_aquifer_chroma_db", tmp_client)
    monkeypatch.setattr(cdb, "openai_ef", DummyEmbeddingFunction())

    # Patch startup to avoid brain initialization
    client = TestClient(create_app(build_default_service_container()))

    # Missing collection -> 404
    resp = client.get(f"{ADMIN_CHROMA_PREFIX}/collection/missing/ids")
    assert resp.status_code == HTTPStatus.NOT_FOUND
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
    resp = client.get(f"{ADMIN_CHROMA_PREFIX}/collection/idscol/ids")
    assert resp.status_code == HTTPStatus.OK
    body = resp.json()
    assert body["collection"] == "idscol"
    assert body["count"] == EXPECTED_COLLECTION_DOCUMENT_COUNT
    assert sorted(body["ids"]) == sorted(["tn_ACT_vcsw", "2", "3"])
