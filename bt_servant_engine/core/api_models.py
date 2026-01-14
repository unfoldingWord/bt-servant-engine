"""API request/response models for the client-agnostic REST API."""

from __future__ import annotations

from datetime import datetime
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

    # Progress messaging fields
    progress_callback_url: str | None = Field(
        default=None,
        description="URL to POST progress messages during processing. "
        "Engine will POST ProgressMessage payloads to this URL.",
    )
    progress_throttle_seconds: float = Field(
        default=3.0,
        ge=0.5,
        le=30.0,
        description="Minimum seconds between progress messages (default: 3.0)",
    )


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


class ProgressMessage(BaseModel):
    """Payload sent to progress_callback_url during processing."""

    user_id: str = Field(..., description="User receiving this progress update")
    message_key: str = Field(..., description="Status message key (e.g., 'REVIEWING_FIA_GUIDANCE')")
    text: str = Field(..., description="Localized progress text to display")
    timestamp: float = Field(..., description="Unix timestamp when progress was sent")


class ChatHistoryEntry(BaseModel):
    """Single chat history entry with timestamp."""

    user_message: str = Field(..., description="User's message")
    assistant_response: str = Field(..., description="Assistant's response")
    created_at: datetime | None = Field(
        default=None,
        description="When this exchange occurred (None for legacy entries without timestamps)",
    )


class ChatHistoryResponse(BaseModel):
    """Response model for GET /api/v1/users/{user_id}/history."""

    user_id: str = Field(..., description="User identifier")
    entries: list[ChatHistoryEntry] = Field(
        default_factory=list,
        description="Chat history entries, newest first",
    )
    total_count: int = Field(..., description="Total number of entries stored")
    limit: int = Field(..., description="Maximum entries returned in this response")
    offset: int = Field(..., description="Number of entries skipped")


__all__ = [
    "ChatRequest",
    "ChatResponse",
    "UserPreferences",
    "ProgressMessage",
    "ChatHistoryEntry",
    "ChatHistoryResponse",
]
