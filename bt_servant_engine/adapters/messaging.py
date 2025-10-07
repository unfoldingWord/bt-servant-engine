"""Messaging adapter implementing the messaging port."""

from __future__ import annotations

from messaging import (
    send_text_message,
    send_typing_indicator_message,
    send_voice_message,
    transcribe_voice_message,
)

from bt_servant_engine.core.ports import MessagingPort


class MessagingAdapter(MessagingPort):
    """Concrete adapter wrapping the legacy messaging helpers."""

    async def send_text_message(self, user_id: str, text: str) -> None:
        await send_text_message(user_id, text)

    async def send_voice_message(self, user_id: str, text: str) -> None:
        await send_voice_message(user_id, text)

    async def send_typing_indicator(self, message_id: str) -> None:
        await send_typing_indicator_message(message_id)

    async def transcribe_voice_message(self, media_id: str) -> str:
        return await transcribe_voice_message(media_id)


__all__ = ["MessagingAdapter"]
