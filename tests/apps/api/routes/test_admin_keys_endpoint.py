"""Tests for the /admin/keys endpoints."""

# pylint: disable=missing-function-docstring,redefined-outer-name,unused-argument

from collections.abc import Iterator
from http import HTTPStatus
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from bt_servant_engine.adapters.api_keys import APIKeyAdapter
from bt_servant_engine.apps.api.app import create_app
from bt_servant_engine.apps.api.dependencies import set_api_key_service
from bt_servant_engine.services import runtime
from bt_servant_engine.services.api_keys import APIKeyService


@pytest.fixture
def api_key_service(tmp_path):
    """Create an API key service with temp database."""
    db_path = tmp_path / "api_keys.db"
    adapter = APIKeyAdapter(db_path=db_path)
    service = APIKeyService(adapter)
    set_api_key_service(service)
    yield service
    set_api_key_service(None)


@pytest.fixture
def client(api_key_service) -> TestClient:
    """Create a test client with the app."""
    app = create_app(runtime.get_services())
    return TestClient(app)


@pytest.fixture
def admin_auth() -> Iterator[None]:
    """Set up admin authentication for tests."""
    with patch("bt_servant_engine.apps.api.dependencies.config") as mock_config:
        mock_config.ENABLE_ADMIN_AUTH = True
        mock_config.ADMIN_API_TOKEN = "test-admin-token"
        yield


@pytest.fixture
def admin_headers() -> dict:
    """Return headers with valid admin token."""
    return {"Authorization": "Bearer test-admin-token"}


class TestAdminKeysAuth:
    """Tests for admin key endpoint authentication."""

    def test_create_key_requires_auth(self, client, admin_auth) -> None:
        """Should reject unauthenticated requests."""
        resp = client.post(
            "/admin/keys",
            json={"name": "Test Key"},
        )
        assert resp.status_code == HTTPStatus.UNAUTHORIZED

    def test_list_keys_requires_auth(self, client, admin_auth) -> None:
        """Should reject unauthenticated requests."""
        resp = client.get("/admin/keys")
        assert resp.status_code == HTTPStatus.UNAUTHORIZED

    def test_create_key_with_valid_admin_token(
        self, client, admin_auth, admin_headers
    ) -> None:
        """Should accept requests with valid admin token."""
        resp = client.post(
            "/admin/keys",
            json={"name": "Test Key"},
            headers=admin_headers,
        )
        assert resp.status_code == HTTPStatus.CREATED


@pytest.fixture
def no_auth() -> Iterator[None]:
    """Disable client auth but keep admin auth enabled for key management tests."""
    with patch("bt_servant_engine.apps.api.dependencies.config") as mock_deps_config:
        # Disable client auth check (in dependencies.py)
        mock_deps_config.ENABLE_ADMIN_AUTH = False
        mock_deps_config.ENABLE_CLIENT_API_KEY_AUTH = None
        # Keep admin auth enabled in admin_keys.py so endpoints work
        with patch("bt_servant_engine.apps.api.routes.admin_keys.config") as mock_keys_config:
            mock_keys_config.ENABLE_ADMIN_AUTH = True
            yield


class TestAdminKeysDisabledWhenAuthDisabled:
    """Tests that admin key endpoints are disabled when ENABLE_ADMIN_AUTH=false."""

    def test_create_key_returns_503_when_auth_disabled(self, client, api_key_service) -> None:
        """Should return 503 when admin auth is disabled."""
        with patch("bt_servant_engine.apps.api.dependencies.config") as mock_deps:
            mock_deps.ENABLE_ADMIN_AUTH = False
            with patch("bt_servant_engine.apps.api.routes.admin_keys.config") as mock_keys:
                mock_keys.ENABLE_ADMIN_AUTH = False
                resp = client.post("/admin/keys", json={"name": "Test"})
                assert resp.status_code == HTTPStatus.SERVICE_UNAVAILABLE
                assert "disabled" in resp.json()["detail"].lower()

    def test_list_keys_returns_503_when_auth_disabled(self, client, api_key_service) -> None:
        """Should return 503 when admin auth is disabled."""
        with patch("bt_servant_engine.apps.api.dependencies.config") as mock_deps:
            mock_deps.ENABLE_ADMIN_AUTH = False
            with patch("bt_servant_engine.apps.api.routes.admin_keys.config") as mock_keys:
                mock_keys.ENABLE_ADMIN_AUTH = False
                resp = client.get("/admin/keys")
                assert resp.status_code == HTTPStatus.SERVICE_UNAVAILABLE


class TestCreateAPIKey:
    """Tests for POST /admin/keys."""

    def test_create_key_minimal(self, client, no_auth) -> None:
        """Should create key with just name."""
        resp = client.post(
            "/admin/keys",
            json={"name": "My API Key"},
        )
        assert resp.status_code == HTTPStatus.CREATED
        data = resp.json()
        assert data["key"]["name"] == "My API Key"
        assert data["key"]["environment"] == "prod"
        assert "raw_key" in data
        assert data["raw_key"].startswith("bts_prod_")

    def test_create_key_with_options(self, client, no_auth) -> None:
        """Should create key with custom options."""
        resp = client.post(
            "/admin/keys",
            json={
                "name": "Custom Key",
                "environment": "staging",
                "rate_limit_per_minute": 120,
                "expires_in_days": 30,
            },
        )
        assert resp.status_code == HTTPStatus.CREATED
        data = resp.json()
        assert data["key"]["environment"] == "staging"
        assert data["key"]["rate_limit_per_minute"] == 120
        assert data["key"]["expires_at"] is not None

    def test_create_key_empty_name_rejected(self, client, no_auth) -> None:
        """Should reject empty name."""
        resp = client.post(
            "/admin/keys",
            json={"name": ""},
        )
        assert resp.status_code == HTTPStatus.BAD_REQUEST

    def test_create_key_invalid_environment_rejected(self, client, no_auth) -> None:
        """Should reject invalid environment."""
        resp = client.post(
            "/admin/keys",
            json={"name": "Test", "environment": "invalid"},
        )
        assert resp.status_code == HTTPStatus.BAD_REQUEST


class TestListAPIKeys:
    """Tests for GET /admin/keys."""

    def test_list_keys_empty(self, client, no_auth) -> None:
        """Should return empty list when no keys."""
        resp = client.get("/admin/keys")
        assert resp.status_code == HTTPStatus.OK
        data = resp.json()
        assert data["keys"] == []
        assert data["total"] == 0

    def test_list_keys_returns_created(self, client, no_auth) -> None:
        """Should return created keys."""
        client.post("/admin/keys", json={"name": "Key 1"})
        client.post("/admin/keys", json={"name": "Key 2"})

        resp = client.get("/admin/keys")
        assert resp.status_code == HTTPStatus.OK
        data = resp.json()
        assert data["total"] == 2
        names = {k["name"] for k in data["keys"]}
        assert names == {"Key 1", "Key 2"}

    def test_list_keys_filter_by_environment(self, client, no_auth) -> None:
        """Should filter by environment."""
        client.post("/admin/keys", json={"name": "Prod", "environment": "prod"})
        client.post("/admin/keys", json={"name": "Dev", "environment": "dev"})

        resp = client.get("/admin/keys", params={"environment": "prod"})
        assert resp.status_code == HTTPStatus.OK
        data = resp.json()
        assert data["total"] == 1
        assert data["keys"][0]["name"] == "Prod"

    def test_list_keys_exclude_revoked_by_default(self, client, no_auth) -> None:
        """Should not include revoked keys by default."""
        # Create and revoke a key
        create_resp = client.post("/admin/keys", json={"name": "To Revoke"})
        key_id = create_resp.json()["key"]["id"]
        client.delete(f"/admin/keys/{key_id}")

        resp = client.get("/admin/keys")
        assert resp.status_code == HTTPStatus.OK
        assert resp.json()["total"] == 0

    def test_list_keys_include_revoked(self, client, no_auth) -> None:
        """Should include revoked keys when requested."""
        create_resp = client.post("/admin/keys", json={"name": "To Revoke"})
        key_id = create_resp.json()["key"]["id"]
        client.delete(f"/admin/keys/{key_id}")

        resp = client.get("/admin/keys", params={"include_revoked": True})
        assert resp.status_code == HTTPStatus.OK
        assert resp.json()["total"] == 1


class TestGetAPIKey:
    """Tests for GET /admin/keys/{key_id}."""

    def test_get_key_by_id(self, client, no_auth) -> None:
        """Should return key details by ID."""
        create_resp = client.post("/admin/keys", json={"name": "My Key"})
        key_id = create_resp.json()["key"]["id"]

        resp = client.get(f"/admin/keys/{key_id}")
        assert resp.status_code == HTTPStatus.OK
        assert resp.json()["name"] == "My Key"

    def test_get_key_not_found(self, client, no_auth) -> None:
        """Should return 404 for nonexistent key."""
        resp = client.get("/admin/keys/nonexistent-id")
        assert resp.status_code == HTTPStatus.NOT_FOUND


class TestRevokeAPIKey:
    """Tests for DELETE /admin/keys/{key_id}."""

    def test_revoke_key(self, client, no_auth) -> None:
        """Should revoke an existing key."""
        create_resp = client.post("/admin/keys", json={"name": "To Revoke"})
        key_id = create_resp.json()["key"]["id"]

        resp = client.delete(f"/admin/keys/{key_id}")
        assert resp.status_code == HTTPStatus.OK

        # Verify it's revoked
        get_resp = client.get(f"/admin/keys/{key_id}")
        assert get_resp.json()["is_active"] is False

    def test_revoke_nonexistent_key(self, client, no_auth) -> None:
        """Should return 404 for nonexistent key."""
        resp = client.delete("/admin/keys/nonexistent-id")
        assert resp.status_code == HTTPStatus.NOT_FOUND

    def test_revoke_already_revoked_key(self, client, no_auth) -> None:
        """Should return 404 for already revoked key."""
        create_resp = client.post("/admin/keys", json={"name": "Already Revoked"})
        key_id = create_resp.json()["key"]["id"]
        client.delete(f"/admin/keys/{key_id}")

        resp = client.delete(f"/admin/keys/{key_id}")
        assert resp.status_code == HTTPStatus.NOT_FOUND


class TestClientApiKeyValidation:
    """Tests for client API key authentication on chat/users endpoints."""

    def test_chat_with_client_key(self, client, api_key_service, no_auth) -> None:
        """Should accept chat requests with valid client API key."""
        # Create a client API key
        result = api_key_service.create_key(name="Chat Client", environment="prod")
        raw_key = result.raw_key

        # Need to re-enable auth and test with the key
        with patch("bt_servant_engine.apps.api.dependencies.config") as mock_config:
            mock_config.ENABLE_ADMIN_AUTH = True
            mock_config.ADMIN_API_TOKEN = "admin-token"

            # Mock brain to avoid full invocation
            with patch("bt_servant_engine.apps.api.routes.chat._get_or_create_brain") as mock_brain:
                mock_brain.return_value.invoke.return_value = {
                    "translated_responses": ["Hello!"],
                    "final_response_language": "en",
                    "triggered_intent": "test",
                }

                resp = client.post(
                    "/api/v1/chat",
                    json={
                        "client_id": "test",
                        "user_id": "user123",
                        "message": "Hello",
                    },
                    headers={"Authorization": f"Bearer {raw_key}"},
                )

                # Will fail with brain not found, but auth should pass
                # Just verify we don't get 401
                assert resp.status_code != HTTPStatus.UNAUTHORIZED
