"""Tests for the /api/v1/users endpoints."""
# pylint: disable=missing-function-docstring

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


class TestUserPreferencesAuth:
    """Tests for /api/v1/users/{user_id}/preferences authentication."""

    def test_missing_auth_header_when_token_configured(
        self, client: TestClient
    ) -> None:
        """Should reject requests without auth header when ADMIN_API_TOKEN is set."""
        with patch(
            "bt_servant_engine.apps.api.routes.users.config"
        ) as mock_config:
            mock_config.ADMIN_API_TOKEN = "test-token"

            resp = client.get("/api/v1/users/user123/preferences")
            assert resp.status_code == HTTPStatus.UNAUTHORIZED

    def test_wrong_api_key(self, client: TestClient) -> None:
        """Should reject requests with wrong API key."""
        with patch(
            "bt_servant_engine.apps.api.routes.users.config"
        ) as mock_config:
            mock_config.ADMIN_API_TOKEN = "correct-token"

            resp = client.get(
                "/api/v1/users/user123/preferences",
                headers={"Authorization": "Bearer wrong-token"},
            )
            assert resp.status_code == HTTPStatus.UNAUTHORIZED


class TestGetUserPreferences:
    """Tests for GET /api/v1/users/{user_id}/preferences."""

    def test_get_preferences_for_new_user(self, client: TestClient) -> None:
        """Should return null/default preferences for new user."""
        resp = client.get("/api/v1/users/new-user-123/preferences")
        assert resp.status_code == HTTPStatus.OK
        data = resp.json()
        assert data["response_language"] is None
        assert data["agentic_strength"] is None
        assert data["dev_agentic_mcp"] is None

    def test_get_preferences_for_existing_user(
        self, client: TestClient, service_container  # noqa: ANN001
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

    def test_update_response_language(self, client: TestClient) -> None:
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

    def test_update_agentic_strength(self, client: TestClient) -> None:
        """Should update agentic strength preference."""
        resp = client.put(
            "/api/v1/users/user-agentic/preferences",
            json={"agentic_strength": "very_low"},
        )
        assert resp.status_code == HTTPStatus.OK
        data = resp.json()
        assert data["agentic_strength"] == "very_low"

    def test_update_dev_agentic_mcp(self, client: TestClient) -> None:
        """Should update dev MCP flag."""
        resp = client.put(
            "/api/v1/users/user-mcp/preferences",
            json={"dev_agentic_mcp": True},
        )
        assert resp.status_code == HTTPStatus.OK
        data = resp.json()
        assert data["dev_agentic_mcp"] is True

    def test_update_multiple_preferences(self, client: TestClient) -> None:
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

    def test_partial_update_preserves_other_fields(
        self, client: TestClient
    ) -> None:
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
