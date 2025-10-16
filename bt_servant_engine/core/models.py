"""Core data transfer objects shared across layers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, List, Mapping, Optional

from pydantic import BaseModel

from bt_servant_engine.core.config import config


@dataclass(slots=True)
class RequestContext:
    """Metadata describing an inbound API request."""

    correlation_id: str
    path: str
    method: str
    user_agent: Optional[str] = None


valid_message_types = {"text", "audio"}


@dataclass(slots=True)
class UserMessage:
    """Normalized user message with basic validation and helpers."""

    message_id: str
    user_id: str
    message_type: str
    timestamp: int
    text: str = ""
    media_id: str = ""

    def __post_init__(self) -> None:
        self.timestamp = int(self.timestamp)

    @classmethod
    def from_data(cls, data: Mapping[str, Any]):
        """Build a UserMessage from a raw Meta webhook message payload."""
        user_id = data.get("from")
        message_id = data.get("id")
        message_type = data.get("type")
        timestamp = data.get("timestamp")

        if not isinstance(user_id, str) or not user_id:
            raise ValueError("missing or invalid sender id")
        if not isinstance(message_id, str) or not message_id:
            raise ValueError("missing or invalid message id")
        if not isinstance(message_type, str) or not message_type:
            raise ValueError("missing or invalid message type")
        if not isinstance(timestamp, str) or not timestamp:
            raise ValueError("missing or invalid timestamp")
        try:
            timestamp_int = int(timestamp)
        except (TypeError, ValueError) as exc:
            raise ValueError("timestamp must be numeric") from exc

        text = ""
        media_id = ""
        if message_type == "text":
            text_field = data.get("text")
            if not isinstance(text_field, Mapping):
                raise ValueError("text message payload malformed: missing text field")
            text = text_field.get("body", "") if isinstance(text_field.get("body", ""), str) else ""
            if not text:
                raise ValueError("text message received with no message body")
        elif message_type == "audio":
            audio_field = data.get("audio")
            if not isinstance(audio_field, Mapping):
                raise ValueError("audio message payload malformed: missing audio field")
            mid = audio_field.get("id")
            if not isinstance(mid, str) or not mid:
                raise ValueError("audio message received with no id")
            media_id = mid

        return cls(
            user_id=user_id,
            message_id=message_id,
            message_type=message_type,
            timestamp=timestamp_int,
            text=text,
            media_id=media_id,
        )

    def age(self) -> float:
        """Return the message age in seconds as a float."""
        message_time_utc = datetime.fromtimestamp(self.timestamp, tz=timezone.utc)
        return (datetime.now(timezone.utc) - message_time_utc).total_seconds()

    def too_old(self) -> bool:
        """Whether the message is older than our accepted threshold."""
        return self.age() > config.MESSAGE_AGE_CUTOFF_IN_SECONDS

    def is_unauthorized_sender(self) -> bool:
        """True when sandbox mode is on and sender is not the sandbox number."""
        return config.IN_META_SANDBOX_MODE and self.user_id != config.META_SANDBOX_PHONE_NUMBER

    def is_supported_type(self) -> bool:
        """True if message_type is currently supported (text or audio)."""
        return self.message_type in valid_message_types


class PassageRef(BaseModel):
    """Normalized reference to a passage within a single canonical book."""

    book: str
    start_chapter: int | None = None
    start_verse: int | None = None
    end_chapter: int | None = None
    end_verse: int | None = None


class PassageSelection(BaseModel):
    """Structured selection consisting of one or more ranges for a book."""

    selections: List[PassageRef]


__all__ = [
    "RequestContext",
    "UserMessage",
    "valid_message_types",
    "PassageRef",
    "PassageSelection",
]
