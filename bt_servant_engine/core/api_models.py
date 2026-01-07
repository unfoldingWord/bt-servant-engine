"""API request/response models for the client-agnostic REST API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Request model for POST /api/v1/chat."""

    client_id: str = Field(
        ..., description="Client identifier (e.g., 'whatsapp', 'web', 'telegram')"
    )
    user_id: str = Field(..., description="User identifier from the client platform")
    message: str = Field(default="", description="Text message (empty if audio)")
    message_type: Literal["text", "audio"] = Field(
        default="text", description="Whether this is a text or audio message"
    )
    audio_base64: str | None = Field(
        default=None, description="Base64-encoded audio data (if message_type='audio')"
    )
    audio_format: str = Field(default="ogg", description="Audio format hint (e.g., 'ogg', 'mp3')")


class ChatResponse(BaseModel):
    """Response model for POST /api/v1/chat."""

    responses: list[str] = Field(
        ..., description="Response texts (includes continuation prompts if queued intents)"
    )
    response_language: str = Field(..., description="Language of the response")
    voice_audio_base64: str | None = Field(
        default=None,
        description="TTS audio as base64 MP3 (if generated) - client decides how to use",
    )
    intent_processed: str = Field(..., description="Which intent was handled")
    has_queued_intents: bool = Field(
        default=False, description="True if more intents are waiting for next request"
    )


class UserPreferences(BaseModel):
    """User preferences model for GET/PUT /api/v1/users/{user_id}/preferences."""

    response_language: str | None = Field(default=None, description="Preferred response language")
    agentic_strength: Literal["normal", "low", "very_low"] | None = Field(
        default=None, description="Agentic behavior strength"
    )
    dev_agentic_mcp: bool | None = Field(default=None, description="Developer MCP flag")


__all__ = ["ChatRequest", "ChatResponse", "UserPreferences"]
