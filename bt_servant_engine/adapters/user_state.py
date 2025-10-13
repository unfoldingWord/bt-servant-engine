"""User state adapter implementing the TinyDB-backed port.

Provides TinyDB database wiring and user data access helpers.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, cast

from tinydb import Query, TinyDB

from bt_servant_engine.core.config import config
from bt_servant_engine.core.ports import UserStatePort

# Provide a QueryLike alias for static checkers; at runtime use Any.
if TYPE_CHECKING:  # pragma: no cover - typing only
    from tinydb.queries import QueryLike  # type: ignore
else:
    QueryLike = Any  # type: ignore[misc,assignment]

# NOTE: consider moving to configuration if needed.
CHAT_HISTORY_MAX = 5
VALID_AGENTIC_STRENGTH = {"normal", "low", "very_low"}

# Resolve from settings dynamically to satisfy static analysis.
DATA_DIR = Path(getattr(config, "DATA_DIR", Path("/data")))
DATA_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH = DATA_DIR / "db.json"
_db = TinyDB(str(DB_PATH))


def get_user_db() -> TinyDB:
    """Return the process-wide TinyDB instance for users."""
    return _db


def get_user_state(user_id: str) -> Dict[str, Any]:
    """Get the entire state dictionary for a user.

    Args:
        user_id: The user's identifier

    Returns:
        Dictionary containing all user state, or empty dict if user not found
    """
    q = Query()
    cond = cast(QueryLike, q.user_id == user_id)
    raw = get_user_db().table("users").get(cond)
    result = cast(Optional[Dict[str, Any]], raw)
    return result.copy() if result else {"user_id": user_id}


def set_user_state(user_id: str, state: Dict[str, Any]) -> None:
    """Set the entire state dictionary for a user.

    Args:
        user_id: The user's identifier
        state: Dictionary containing all user state to persist

    Note:
        This will merge the provided state with existing state, not replace it.
        To remove a field, explicitly set it to None and handle removal separately.
    """
    q = Query()
    db = get_user_db().table("users")
    cond = cast(QueryLike, q.user_id == user_id)

    # Ensure user_id is in the state
    state["user_id"] = user_id

    # Upsert the entire state
    db.upsert(state, cond)


def get_user_chat_history(user_id: str) -> List[Dict[str, str]]:
    """Retrieve chat history for the given user_id."""
    q = Query()
    cond = cast(QueryLike, q.user_id == user_id)
    raw = get_user_db().table("users").get(cond)
    result = cast(Optional[Dict[str, Any]], raw)
    history = result.get("history", []) if result else []
    return cast(List[Dict[str, str]], history)


def update_user_chat_history(user_id: str, query: str, response: str) -> None:
    """Append a user/bot exchange to chat history (bounded)."""
    q = Query()
    users_table = get_user_db().table("users")
    cond = cast(QueryLike, q.user_id == user_id)
    existing_raw = users_table.get(cond)
    existing = cast(Optional[Dict[str, Any]], existing_raw)
    history = cast(List[Dict[str, str]], existing.get("history", []) if existing else [])

    history.append({"user_message": query, "assistant_response": response})

    history = history[-CHAT_HISTORY_MAX:]
    users_table.upsert({"user_id": user_id, "history": history}, cond)


def get_user_response_language(user_id: str) -> Optional[str]:
    """Get the user's preferred response language, or None if not set."""
    q = Query()
    cond = cast(QueryLike, q.user_id == user_id)
    raw = get_user_db().table("users").get(cond)
    result = cast(Optional[Dict[str, Any]], raw)
    lang = result.get("response_language") if result else None
    return cast(Optional[str], lang)


def set_user_response_language(user_id: str, language: str) -> None:
    """Set the user's preferred response language."""
    q = Query()
    db = get_user_db().table("users")
    cond = cast(QueryLike, q.user_id == user_id)
    existing_raw = db.get(cond)
    existing = cast(Optional[Dict[str, Any]], existing_raw)
    updated: Dict[str, Any] = (
        existing.copy() if isinstance(existing, dict) else {"user_id": user_id}
    )

    updated["response_language"] = language
    db.upsert(updated, cond)


def get_user_agentic_strength(user_id: str) -> Optional[str]:
    """Get the user's preferred agentic strength, or None if not set."""
    q = Query()
    cond = cast(QueryLike, q.user_id == user_id)
    raw = get_user_db().table("users").get(cond)
    result = cast(Optional[Dict[str, Any]], raw)
    strength = result.get("agentic_strength") if result else None
    if isinstance(strength, str) and strength in VALID_AGENTIC_STRENGTH:
        return strength
    return None


def set_user_agentic_strength(user_id: str, strength: str) -> None:
    """Persist the user's preferred agentic strength level."""
    normalized = strength.strip().lower()
    if normalized not in VALID_AGENTIC_STRENGTH:
        raise ValueError(f"invalid agentic strength: {strength}")

    q = Query()
    db = get_user_db().table("users")
    cond = cast(QueryLike, q.user_id == user_id)
    existing_raw = db.get(cond)
    existing = cast(Optional[Dict[str, Any]], existing_raw)
    updated: Dict[str, Any] = (
        existing.copy() if isinstance(existing, dict) else {"user_id": user_id}
    )
    updated["agentic_strength"] = normalized
    db.upsert(updated, cond)


def set_first_interaction(user_id: str, is_first: bool) -> None:
    """Set whether this is the user's first interaction."""
    q = Query()
    db = get_user_db().table("users")
    cond = cast(QueryLike, q.user_id == user_id)
    existing_raw = db.get(cond)
    existing = cast(Optional[Dict[str, Any]], existing_raw)
    updated: Dict[str, Any] = (
        existing.copy() if isinstance(existing, dict) else {"user_id": user_id}
    )

    updated["first_interaction"] = is_first
    db.upsert(updated, cond)


def is_first_interaction(user_id: str) -> bool:
    """Return True if this is the user's first interaction, otherwise False."""
    q = Query()
    cond = cast(QueryLike, q.user_id == user_id)
    raw = get_user_db().table("users").get(cond)
    result = cast(Optional[Dict[str, Any]], raw)
    first = (result or {}).get("first_interaction", True)
    return bool(first)


class UserStateAdapter(UserStatePort):
    """Concrete adapter wrapping TinyDB helper functions."""

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


__all__ = [
    "UserStateAdapter",
    "get_user_db",
    "get_user_state",
    "set_user_state",
    "get_user_chat_history",
    "update_user_chat_history",
    "get_user_response_language",
    "set_user_response_language",
    "get_user_agentic_strength",
    "set_user_agentic_strength",
    "set_first_interaction",
    "is_first_interaction",
]
