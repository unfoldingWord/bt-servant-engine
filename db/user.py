"""User data access helpers backed by TinyDB."""

from typing import Any, List, Dict, Optional, cast
from tinydb import Query
from db.user_db import get_user_db

# NOTE: consider moving to configuration if needed.
CHAT_HISTORY_MAX = 5


def get_user_chat_history(user_id: str) -> List[Dict[str, str]]:
    """Retrieve chat history for the given user_id."""
    q = Query()
    raw = get_user_db().table("users").get(q.user_id == user_id)
    result = cast(Optional[Dict[str, Any]], raw)
    history = result.get("history", []) if result else []
    return cast(List[Dict[str, str]], history)


def update_user_chat_history(user_id: str, query: str, response: str) -> None:
    """Append a user/bot exchange to chat history (bounded)."""
    q = Query()
    users_table = get_user_db().table("users")
    existing_raw = users_table.get(q.user_id == user_id)
    existing = cast(Optional[Dict[str, Any]], existing_raw)
    history = cast(List[Dict[str, str]], existing.get("history", []) if existing else [])

    history.append({
        "user_message": query,
        "assistant_response": response
    })

    history = history[-CHAT_HISTORY_MAX:]
    users_table.upsert({"user_id": user_id, "history": history}, q.user_id == user_id)


def get_user_response_language(user_id: str) -> Optional[str]:
    """Get the user's preferred response language, or None if not set."""
    q = Query()
    raw = get_user_db().table("users").get(q.user_id == user_id)
    result = cast(Optional[Dict[str, Any]], raw)
    lang = result.get("response_language") if result else None
    return cast(Optional[str], lang)


def set_user_response_language(user_id: str, language: str) -> None:
    """Set the user's preferred response language."""
    q = Query()
    db = get_user_db().table("users")
    existing_raw = db.get(q.user_id == user_id)
    existing = cast(Optional[Dict[str, Any]], existing_raw)
    updated: Dict[str, Any] = (
        existing.copy() if isinstance(existing, dict) else {"user_id": user_id}
    )

    updated["response_language"] = language
    db.upsert(updated, q.user_id == user_id)


def set_first_interaction(user_id: str, is_first: bool) -> None:
    """Set whether this is the user's first interaction."""
    q = Query()
    db = get_user_db().table("users")
    existing_raw = db.get(q.user_id == user_id)
    existing = cast(Optional[Dict[str, Any]], existing_raw)
    updated: Dict[str, Any] = (
        existing.copy() if isinstance(existing, dict) else {"user_id": user_id}
    )

    updated["first_interaction"] = is_first
    db.upsert(updated, q.user_id == user_id)


def is_first_interaction(user_id: str) -> bool:
    """Return True if this is the user's first interaction, otherwise False."""
    q = Query()
    raw = get_user_db().table("users").get(q.user_id == user_id)
    result = cast(Optional[Dict[str, Any]], raw)
    first = (result or {}).get("first_interaction", True)
    return bool(first)
