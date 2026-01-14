"""Tests for the /api/v1/users endpoints."""
# pylint: disable=missing-function-docstring,redefined-outer-name,unused-argument
# ruff: noqa: PLR2004

from collections.abc import Iterator
from http import HTTPStatus
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from bt_servant_engine.apps.api.app import create_app
from bt_servant_engine.services import runtime


@pytest.fixture
def client() -> TestClient:
    """Create a test client with the app."""
    app = create_app(runtime.get_services())
    return TestClient(app)


@pytest.fixture
def no_auth() -> Iterator[None]:
    """Disable auth for tests that don't test authentication."""
    with patch("bt_servant_engine.apps.api.dependencies.config") as mock_deps_config:
        mock_deps_config.ENABLE_ADMIN_AUTH = False
        with patch("bt_servant_engine.apps.api.routes.users.config") as mock_config:
            mock_config.ADMIN_API_TOKEN = ""
            yield


class TestUserPreferencesAuth:
    """Tests for /api/v1/users/{user_id}/preferences authentication."""

    def test_missing_auth_header_when_token_configured(self, client: TestClient) -> None:
        """Should reject requests without auth header when ADMIN_API_TOKEN is set."""
        with patch("bt_servant_engine.apps.api.routes.users.config") as mock_config:
            mock_config.ADMIN_API_TOKEN = "test-token"

            resp = client.get("/api/v1/users/user123/preferences")
            assert resp.status_code == HTTPStatus.UNAUTHORIZED

    def test_wrong_api_key(self, client: TestClient) -> None:
        """Should reject requests with wrong API key."""
        with patch("bt_servant_engine.apps.api.routes.users.config") as mock_config:
            mock_config.ADMIN_API_TOKEN = "correct-token"

            resp = client.get(
                "/api/v1/users/user123/preferences",
                headers={"Authorization": "Bearer wrong-token"},
            )
            assert resp.status_code == HTTPStatus.UNAUTHORIZED


class TestGetUserPreferences:
    """Tests for GET /api/v1/users/{user_id}/preferences."""

    def test_get_preferences_for_new_user(self, client: TestClient, no_auth: None) -> None:
        """Should return null/default preferences for new user."""
        resp = client.get("/api/v1/users/new-user-123/preferences")
        assert resp.status_code == HTTPStatus.OK
        data = resp.json()
        assert data["response_language"] is None
        assert data["agentic_strength"] is None
        assert data["dev_agentic_mcp"] is None

    def test_get_preferences_for_existing_user(
        self,
        client: TestClient,
        service_container,
        no_auth: None,  # noqa: ANN001
    ) -> None:
        """Should return stored preferences for existing user."""
        # Set up user state
        user_state = service_container.user_state
        user_state.set_response_language(user_id="user456", language="es")
        user_state.set_agentic_strength(user_id="user456", strength="low")
        user_state.set_dev_agentic_mcp(user_id="user456", enabled=True)

        resp = client.get("/api/v1/users/user456/preferences")
        assert resp.status_code == HTTPStatus.OK
        data = resp.json()
        assert data["response_language"] == "es"
        assert data["agentic_strength"] == "low"
        assert data["dev_agentic_mcp"] is True


class TestUpdateUserPreferences:
    """Tests for PUT /api/v1/users/{user_id}/preferences."""

    def test_update_response_language(self, client: TestClient, no_auth: None) -> None:
        """Should update response language preference."""
        resp = client.put(
            "/api/v1/users/user789/preferences",
            json={"response_language": "fr"},
        )
        assert resp.status_code == HTTPStatus.OK
        data = resp.json()
        assert data["response_language"] == "fr"

        # Verify it persisted
        resp2 = client.get("/api/v1/users/user789/preferences")
        assert resp2.json()["response_language"] == "fr"

    def test_update_agentic_strength(self, client: TestClient, no_auth: None) -> None:
        """Should update agentic strength preference."""
        resp = client.put(
            "/api/v1/users/user-agentic/preferences",
            json={"agentic_strength": "very_low"},
        )
        assert resp.status_code == HTTPStatus.OK
        data = resp.json()
        assert data["agentic_strength"] == "very_low"

    def test_update_dev_agentic_mcp(self, client: TestClient, no_auth: None) -> None:
        """Should update dev MCP flag."""
        resp = client.put(
            "/api/v1/users/user-mcp/preferences",
            json={"dev_agentic_mcp": True},
        )
        assert resp.status_code == HTTPStatus.OK
        data = resp.json()
        assert data["dev_agentic_mcp"] is True

    def test_update_multiple_preferences(self, client: TestClient, no_auth: None) -> None:
        """Should update multiple preferences at once."""
        resp = client.put(
            "/api/v1/users/user-multi/preferences",
            json={
                "response_language": "de",
                "agentic_strength": "normal",
                "dev_agentic_mcp": False,
            },
        )
        assert resp.status_code == HTTPStatus.OK
        data = resp.json()
        assert data["response_language"] == "de"
        assert data["agentic_strength"] == "normal"
        assert data["dev_agentic_mcp"] is False

    def test_partial_update_preserves_other_fields(self, client: TestClient, no_auth: None) -> None:
        """Should only update provided fields, preserving others."""
        # First set all preferences
        client.put(
            "/api/v1/users/user-partial/preferences",
            json={
                "response_language": "it",
                "agentic_strength": "low",
            },
        )

        # Then update only one
        resp = client.put(
            "/api/v1/users/user-partial/preferences",
            json={"agentic_strength": "normal"},
        )
        assert resp.status_code == HTTPStatus.OK
        data = resp.json()
        # response_language should still be "it"
        assert data["response_language"] == "it"
        assert data["agentic_strength"] == "normal"


class TestGetUserHistory:
    """Tests for GET /api/v1/users/{user_id}/history."""

    def test_get_history_empty(self, client: TestClient, no_auth: None) -> None:
        """Should return empty history for new user."""
        resp = client.get("/api/v1/users/new-user-history/history")
        assert resp.status_code == HTTPStatus.OK
        data = resp.json()
        assert data["user_id"] == "new-user-history"
        assert data["entries"] == []
        assert data["total_count"] == 0
        assert data["limit"] == 50
        assert data["offset"] == 0

    def test_get_history_with_entries(
        self,
        client: TestClient,
        service_container,  # noqa: ANN001
        no_auth: None,
    ) -> None:
        """Should return stored history entries with timestamps."""
        user_state = service_container.user_state

        # Add some history
        user_state.append_chat_history("user-hist", "Hello", "Hi there!")
        user_state.append_chat_history("user-hist", "How are you?", "I'm fine!")

        resp = client.get("/api/v1/users/user-hist/history")
        assert resp.status_code == HTTPStatus.OK
        data = resp.json()
        assert data["total_count"] == 2
        assert len(data["entries"]) == 2

        # Newest first
        assert data["entries"][0]["user_message"] == "How are you?"
        assert data["entries"][0]["assistant_response"] == "I'm fine!"
        assert data["entries"][1]["user_message"] == "Hello"
        assert data["entries"][1]["assistant_response"] == "Hi there!"

        # Check timestamps exist
        assert data["entries"][0]["created_at"] is not None
        assert data["entries"][1]["created_at"] is not None

    def test_get_history_pagination(
        self,
        client: TestClient,
        service_container,  # noqa: ANN001
        no_auth: None,
    ) -> None:
        """Should support pagination parameters."""
        user_state = service_container.user_state

        # Add 10 entries
        for i in range(10):
            user_state.append_chat_history(
                "user-paginate", f"Message {i}", f"Response {i}"
            )

        # Get with limit and offset
        resp = client.get("/api/v1/users/user-paginate/history?limit=3&offset=2")
        assert resp.status_code == HTTPStatus.OK
        data = resp.json()
        assert data["total_count"] == 10
        assert data["limit"] == 3
        assert data["offset"] == 2
        assert len(data["entries"]) == 3
        # Newest first: 9,8,7,6,5,4,3,2,1,0 -> skip 2 -> 7,6,5
        assert data["entries"][0]["user_message"] == "Message 7"
        assert data["entries"][1]["user_message"] == "Message 6"
        assert data["entries"][2]["user_message"] == "Message 5"

    def test_get_history_limit_clamped(
        self,
        client: TestClient,
        no_auth: None,
    ) -> None:
        """Should clamp limit to 100 max."""
        resp = client.get("/api/v1/users/user-clamp/history?limit=200")
        assert resp.status_code == HTTPStatus.OK
        data = resp.json()
        assert data["limit"] == 100

    def test_get_history_legacy_entries_null_timestamp(
        self,
        client: TestClient,
        service_container,  # noqa: ANN001
        no_auth: None,
    ) -> None:
        """Legacy entries without timestamps should show created_at as null."""
        # Manually insert entry without timestamp
        service_container.user_state.save_user_state(
            "user-legacy",
            {"history": [{"user_message": "old", "assistant_response": "reply"}]},
        )

        resp = client.get("/api/v1/users/user-legacy/history")
        assert resp.status_code == HTTPStatus.OK
        data = resp.json()
        assert len(data["entries"]) == 1
        assert data["entries"][0]["user_message"] == "old"
        assert data["entries"][0]["created_at"] is None
