"""Protocol definitions for infrastructure adapters."""

# pylint: disable=unnecessary-ellipsis

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Protocol

if TYPE_CHECKING:
    from bt_servant_engine.core.api_key_models import APIKey


class ChromaPort(Protocol):
    """Port exposing collection storage operations backed by Chroma."""

    def get_collection(self, name: str) -> Any | None:
        """Return an existing collection handle or ``None`` if missing."""
        ...

    def list_collections(self) -> list[str]:
        """Return the available Chroma collection names."""
        ...

    def create_collection(self, name: str) -> None:
        """Create a new collection with ``name``."""
        ...

    def delete_collection(self, name: str) -> None:
        """Delete the collection identified by ``name``."""
        ...

    def delete_document(self, name: str, document_id: str) -> None:
        """Delete a single document by id from ``name``."""
        ...

    def count_documents(self, name: str) -> int:
        """Return the number of documents stored in ``name``."""
        ...

    def get_document_text_and_metadata(
        self, name: str, document_id: str
    ) -> tuple[str, Mapping[str, Any]]:
        """Return the text and metadata for a specific document."""
        ...

    def list_document_ids(self, name: str) -> list[str]:
        """Return the document ids for ``name``."""
        ...

    def iter_batches(
        self,
        name: str,
        *,
        batch_size: int = 1000,
        include_embeddings: bool = False,
    ) -> Iterable[dict[str, Any]]:
        """Yield batches of items from ``name``."""
        ...

    def get_collections_pair(self, source: str, dest: str) -> tuple[Any, Any]:
        """Return the source/dest collection handles for a merge."""
        ...

    def max_numeric_id(self, name: str) -> int:
        """Return the maximum numeric document id stored in ``name``."""
        ...


class UserStatePort(Protocol):
    """Port exposing user preference and history persistence."""

    def load_user_state(self, user_id: str) -> Mapping[str, Any]:
        """Return the entire persisted state dictionary for ``user_id``."""
        ...

    def save_user_state(self, user_id: str, state: Mapping[str, Any]) -> None:
        """Persist the provided state dictionary for ``user_id``."""
        ...

    def get_chat_history(self, user_id: str) -> list[dict[str, Any]]:
        """Return the full stored chat history for ``user_id``.

        Returns entries with keys: user_message, assistant_response, created_at (optional).
        """
        ...

    def get_chat_history_for_llm(self, user_id: str) -> list[dict[str, str]]:
        """Return truncated history for LLM context (respects CHAT_HISTORY_LLM_MAX).

        Returns only user_message and assistant_response, no timestamps.
        """
        ...

    def append_chat_history(
        self,
        user_id: str,
        query: str,
        response: str,
        created_at: datetime | None = None,
    ) -> None:
        """Append a turn to the stored chat history with optional timestamp."""
        ...

    def get_response_language(self, user_id: str) -> str | None:
        """Return the preferred response language if stored."""
        ...

    def set_response_language(self, user_id: str, language: str) -> None:
        """Persist the response language preference."""
        ...

    def clear_response_language(self, user_id: str) -> None:
        """Remove any stored response language preference."""
        ...

    def get_last_response_language(self, user_id: str) -> str | None:
        """Return the last response language sent to ``user_id`` (if stored)."""
        ...

    def set_last_response_language(self, user_id: str, language: str) -> None:
        """Persist the last response language sent to ``user_id``."""
        ...

    def get_agentic_strength(self, user_id: str) -> str | None:
        """Return the stored agentic strength preference."""
        ...

    def set_agentic_strength(self, user_id: str, strength: str) -> None:
        """Persist the agentic strength preference."""
        ...

    def get_dev_agentic_mcp(self, user_id: str) -> bool | None:
        """Return whether the dev MCP agentic mode is enabled for the user."""
        ...

    def set_dev_agentic_mcp(self, user_id: str, enabled: bool) -> None:
        """Persist the dev MCP agentic mode preference."""
        ...

    def set_first_interaction(self, user_id: str, is_first: bool) -> None:
        """Mark whether ``user_id`` is on their first interaction."""
        ...

    def is_first_interaction(self, user_id: str) -> bool:
        """Return whether ``user_id`` is on their first interaction."""
        ...


class APIKeyPort(Protocol):
    """Port for API key storage and validation operations."""

    def create_key(
        self,
        name: str,
        environment: str,
        rate_limit_per_minute: int = 60,
        expires_at: datetime | None = None,
    ) -> tuple["APIKey", str]:
        """Create a new API key. Returns (key_metadata, raw_key)."""
        ...

    def validate_key(self, raw_key: str) -> "APIKey | None":
        """Validate a raw key and return metadata if valid, None otherwise."""
        ...

    def get_key_by_id(self, key_id: str) -> "APIKey | None":
        """Get key metadata by ID."""
        ...

    def get_key_by_prefix(self, prefix: str) -> "APIKey | None":
        """Get key metadata by prefix (for admin lookup)."""
        ...

    def list_keys(
        self,
        include_revoked: bool = False,
        environment: str | None = None,
    ) -> list["APIKey"]:
        """List all keys matching criteria."""
        ...

    def revoke_key(self, key_id: str) -> bool:
        """Revoke a key. Returns True if successful, False if not found."""
        ...

    def update_last_used(self, key_id: str) -> None:
        """Update the last_used_at timestamp for a key."""
        ...


__all__ = ["ChromaPort", "UserStatePort", "APIKeyPort"]
