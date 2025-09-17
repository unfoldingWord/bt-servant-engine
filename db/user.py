"""User data access helpers backed by TinyDB."""

from typing import Any, List, Dict, Optional, cast, TYPE_CHECKING
from tinydb import Query
from db.user_db import get_user_db

# Provide a QueryLike alias for static checkers; at runtime use Any.
if TYPE_CHECKING:  # pragma: no cover - typing only
    from tinydb.queries import QueryLike  # type: ignore
else:
    QueryLike = Any  # type: ignore[misc,assignment]

# NOTE: consider moving to configuration if needed.
CHAT_HISTORY_MAX = 5
VALID_AGENTIC_STRENGTH = {"normal", "low", "very_low"}


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

    history.append({
        "user_message": query,
        "assistant_response": response
    })

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
