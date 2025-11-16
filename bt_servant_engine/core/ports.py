"""Protocol definitions for infrastructure adapters."""

# pylint: disable=unnecessary-ellipsis

from __future__ import annotations

from typing import Any, Iterable, Mapping, Protocol


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

    def get_chat_history(self, user_id: str) -> list[dict[str, str]]:
        """Return the recent chat history for ``user_id``."""
        ...

    def append_chat_history(self, user_id: str, query: str, response: str) -> None:
        """Append a turn to the stored chat history."""
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

    def set_first_interaction(self, user_id: str, is_first: bool) -> None:
        """Mark whether ``user_id`` is on their first interaction."""
        ...

    def is_first_interaction(self, user_id: str) -> bool:
        """Return whether ``user_id`` is on their first interaction."""
        ...


class MessagingPort(Protocol):
    """Port exposing messaging-related side effects."""

    async def send_text_message(self, user_id: str, text: str) -> None:
        """Send a plain text message to ``user_id``."""
        ...

    async def send_voice_message(self, user_id: str, text: str) -> None:
        """Send synthesized audio to ``user_id``."""
        ...

    async def send_typing_indicator(self, message_id: str) -> None:
        """Emit a typing indicator for ``message_id``."""
        ...

    async def transcribe_voice_message(self, media_id: str) -> str:
        """Return the transcribed text for a Meta media id."""
        ...


__all__ = ["ChromaPort", "UserStatePort", "MessagingPort"]
