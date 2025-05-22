from db.user_db import get_user_db
from tinydb import Query
from typing import List, Dict, Optional

CHAT_HISTORY_MAX = 5


def get_user_chat_history(user_id: str) -> List[Dict[str, str]]:
    """Retrieve chat history for the given user_id."""
    q = Query()
    result = get_user_db().table("users").get(q.user_id == user_id)
    return result.get("history", []) if result else []


def update_user_chat_history(user_id: str, query: str, response: str) -> None:
    q = Query()
    users_table = get_user_db().table("users")
    existing = users_table.get(q.user_id == user_id)
    history = existing.get("history", []) if existing else []

    history.append({
        "user_message": query,
        "assistant_response": response
    })

    history = history[-CHAT_HISTORY_MAX:]
    users_table.upsert({"user_id": user_id, "history": history}, q.user_id == user_id)


def get_user_response_language(user_id: str) -> Optional[str]:
    """Get the user's preferred response language, or None if not set."""
    q = Query()
    result = get_user_db().table("users").get(q.user_id == user_id)
    return result.get("response_language") if result else None


def set_user_response_language(user_id: str, language: str) -> None:
    """Set the user's preferred response language."""
    q = Query()
    db = get_user_db().table("users")

    existing = db.get(q.user_id == user_id)
    updated = existing.copy() if existing else {"user_id": user_id}

    updated["response_language"] = language
    db.upsert(updated, q.user_id == user_id)
