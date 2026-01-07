"""Tests for the /api/v1/chat endpoint."""
# pylint: disable=missing-function-docstring,redefined-outer-name,too-few-public-methods,unused-argument

from http import HTTPStatus
from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from bt_servant_engine.apps.api.app import create_app, set_brain
from bt_servant_engine.services import runtime


@pytest.fixture
def client() -> TestClient:
    """Create a test client with the app."""
    app = create_app(runtime.get_services())
    return TestClient(app)


@pytest.fixture
def mock_brain() -> Iterator[MagicMock]:
    """Create a mock brain that returns a simple response."""
    brain = MagicMock()
    brain.invoke.return_value = {
        "translated_responses": ["Hello! How can I help you today?"],
        "final_response_language": "en",
        "triggered_intent": "general-conversation",
        "send_voice_message": False,
    }
    set_brain(brain)
    yield brain
    set_brain(None)


class TestChatEndpointAuth:
    """Tests for /api/v1/chat authentication."""

    def test_missing_auth_header_when_token_configured(
        self, client: TestClient
    ) -> None:
        """Should reject requests without auth header when ADMIN_API_TOKEN is set."""
        with patch(
            "bt_servant_engine.apps.api.routes.chat.config"
        ) as mock_config:
            mock_config.ADMIN_API_TOKEN = "test-token"
            mock_config.OPENAI_API_KEY = "test-key"
            mock_config.AGENTIC_STRENGTH = "normal"
            mock_config.BT_DEV_AGENTIC_MCP = False

            resp = client.post(
                "/api/v1/chat",
                json={
                    "client_id": "test",
                    "user_id": "user123",
                    "message": "Hello",
                },
            )
            assert resp.status_code == HTTPStatus.UNAUTHORIZED

    def test_invalid_auth_header_format(self, client: TestClient) -> None:
        """Should reject requests with invalid auth header format."""
        with patch(
            "bt_servant_engine.apps.api.routes.chat.config"
        ) as mock_config:
            mock_config.ADMIN_API_TOKEN = "test-token"

            resp = client.post(
                "/api/v1/chat",
                json={
                    "client_id": "test",
                    "user_id": "user123",
                    "message": "Hello",
                },
                headers={"Authorization": "InvalidFormat"},
            )
            assert resp.status_code == HTTPStatus.UNAUTHORIZED

    def test_wrong_api_key(self, client: TestClient) -> None:
        """Should reject requests with wrong API key."""
        with patch(
            "bt_servant_engine.apps.api.routes.chat.config"
        ) as mock_config:
            mock_config.ADMIN_API_TOKEN = "correct-token"

            resp = client.post(
                "/api/v1/chat",
                json={
                    "client_id": "test",
                    "user_id": "user123",
                    "message": "Hello",
                },
                headers={"Authorization": "Bearer wrong-token"},
            )
            assert resp.status_code == HTTPStatus.UNAUTHORIZED


@pytest.fixture
def no_auth() -> Iterator[None]:
    """Disable auth for tests that don't test authentication."""
    with patch("bt_servant_engine.apps.api.routes.chat.config") as mock_config:
        mock_config.ADMIN_API_TOKEN = ""
        mock_config.OPENAI_API_KEY = "test-key"
        mock_config.AGENTIC_STRENGTH = "normal"
        mock_config.BT_DEV_AGENTIC_MCP = False
        yield


class TestChatEndpointValidation:
    """Tests for /api/v1/chat request validation."""

    def test_empty_message_rejected(
        self, client: TestClient, mock_brain: MagicMock, no_auth: None
    ) -> None:
        """Should reject requests with empty message."""
        resp = client.post(
            "/api/v1/chat",
            json={
                "client_id": "test",
                "user_id": "user123",
                "message": "",
            },
        )
        assert resp.status_code == HTTPStatus.BAD_REQUEST
        assert "empty" in resp.json()["detail"].lower()

    def test_audio_without_audio_data_rejected(
        self, client: TestClient, mock_brain: MagicMock, no_auth: None
    ) -> None:
        """Should reject audio message_type without audio_base64."""
        resp = client.post(
            "/api/v1/chat",
            json={
                "client_id": "test",
                "user_id": "user123",
                "message": "",
                "message_type": "audio",
            },
        )
        assert resp.status_code == HTTPStatus.BAD_REQUEST
        assert "audio_base64" in resp.json()["detail"].lower()


class TestChatEndpointSuccess:
    """Tests for successful /api/v1/chat requests."""

    def test_text_message_success(
        self, client: TestClient, mock_brain: MagicMock, no_auth: None
    ) -> None:
        """Should process text message and return response."""
        resp = client.post(
            "/api/v1/chat",
            json={
                "client_id": "whatsapp",
                "user_id": "user123",
                "message": "Hello, how are you?",
            },
        )
        assert resp.status_code == HTTPStatus.OK
        data = resp.json()
        assert "responses" in data
        assert len(data["responses"]) > 0
        assert data["response_language"] == "en"
        assert data["intent_processed"] == "general-conversation"
        assert data["has_queued_intents"] is False

    def test_brain_invoked_with_correct_payload(
        self, client: TestClient, mock_brain: MagicMock, no_auth: None
    ) -> None:
        """Should invoke brain with correct payload structure."""
        client.post(
            "/api/v1/chat",
            json={
                "client_id": "whatsapp",
                "user_id": "user456",
                "message": "Show me Romans 8",
            },
        )

        mock_brain.invoke.assert_called_once()
        payload = mock_brain.invoke.call_args[0][0]
        assert payload["user_id"] == "user456"
        assert payload["user_query"] == "Show me Romans 8"
        assert "agentic_strength" in payload
        assert payload["progress_enabled"] is False


class TestChatEndpointVoice:
    """Tests for voice-related functionality."""

    def test_voice_response_when_brain_requests_it(
        self, client: TestClient, no_auth: None
    ) -> None:
        """Should include voice audio when brain requests voice output."""
        brain = MagicMock()
        brain.invoke.return_value = {
            "translated_responses": ["Here is your scripture."],
            "final_response_language": "en",
            "triggered_intent": "retrieve-scripture",
            "send_voice_message": True,
            "voice_message_text": "Here is your scripture reading.",
        }
        set_brain(brain)

        with patch(
            "bt_servant_engine.apps.api.routes.chat._generate_tts"
        ) as mock_tts:
            mock_tts.return_value = "base64encodedaudio=="

            resp = client.post(
                "/api/v1/chat",
                json={
                    "client_id": "whatsapp",
                    "user_id": "user123",
                    "message": "Read Romans 8 to me",
                },
            )

            assert resp.status_code == HTTPStatus.OK
            data = resp.json()
            assert data["voice_audio_base64"] == "base64encodedaudio=="
            mock_tts.assert_called_once()

        set_brain(None)
