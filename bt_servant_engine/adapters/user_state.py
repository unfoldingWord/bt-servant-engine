"""Adapter that implements the user state port with TinyDB helpers."""

from __future__ import annotations

from db import (
    get_user_agentic_strength,
    get_user_chat_history,
    get_user_response_language,
    is_first_interaction,
    set_first_interaction,
    set_user_agentic_strength,
    set_user_response_language,
    update_user_chat_history,
)

from bt_servant_engine.core.ports import UserStatePort


class UserStateAdapter(UserStatePort):
    """Concrete ``UserStatePort`` using the legacy TinyDB helpers."""

    def get_chat_history(self, user_id: str) -> list[dict[str, str]]:
        return get_user_chat_history(user_id)

    def append_chat_history(self, user_id: str, query: str, response: str) -> None:
        update_user_chat_history(user_id, query, response)

    def get_response_language(self, user_id: str) -> str | None:
        return get_user_response_language(user_id)

    def set_response_language(self, user_id: str, language: str) -> None:
        set_user_response_language(user_id, language)

    def get_agentic_strength(self, user_id: str) -> str | None:
        return get_user_agentic_strength(user_id)

    def set_agentic_strength(self, user_id: str, strength: str) -> None:
        set_user_agentic_strength(user_id, strength)

    def set_first_interaction(self, user_id: str, is_first: bool) -> None:
        set_first_interaction(user_id, is_first)

    def is_first_interaction(self, user_id: str) -> bool:
        return is_first_interaction(user_id)


__all__ = ["UserStateAdapter"]
