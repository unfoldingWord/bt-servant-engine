"""DB package public interface: user and Chroma helpers."""

from .user_db import get_user_db
from .chroma_db import (
    get_or_create_chroma_collection,
    get_chroma_collection,
)
from .user import (
    get_user_chat_history,
    update_user_chat_history,
    get_user_response_language,
    set_user_response_language,
    set_first_interaction,
    is_first_interaction
)
__all__ = [
    "get_user_chat_history",
    "update_user_chat_history",
    "get_user_response_language",
    "set_user_response_language",
    "set_first_interaction",
    "is_first_interaction",
    "get_or_create_chroma_collection",
    "get_chroma_collection",
]
