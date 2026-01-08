"""Pytest configuration: ensure env vars and import path are set early.

This runs before any tests, so modules can import without local path hacks.
Also load .env before setting defaults so real keys are used when present.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Iterable, Iterator, Mapping

import pytest
from dotenv import load_dotenv

from bt_servant_engine.core.ports import ChromaPort, UserStatePort
from bt_servant_engine.services import ServiceContainer, runtime
from bt_servant_engine.services.intent_router import IntentRouter

# Load .env first so OPENAI_API_KEY and friends are available for tests
load_dotenv(override=False)

# Ensure required env vars exist for config import in app (fallbacks only)
os.environ.setdefault("OPENAI_API_KEY", "test")
os.environ.setdefault("BASE_URL", "http://example.com")
os.environ.setdefault("LOG_PSEUDONYM_SECRET", "test-secret")
os.environ.setdefault("ENABLE_ADMIN_AUTH", "false")


class InMemoryUserStatePort(UserStatePort):
    """Simple in-memory user state store for tests."""

    def __init__(self) -> None:
        self._states: Dict[str, Dict[str, Any]] = {}

    def load_user_state(self, user_id: str) -> Mapping[str, Any]:
        return self._states.get(user_id, {"user_id": user_id}).copy()

    def save_user_state(self, user_id: str, state: Mapping[str, Any]) -> None:
        current = self._states.get(user_id, {"user_id": user_id}).copy()
        current.update(dict(state))
        self._states[user_id] = current

    def get_chat_history(self, user_id: str) -> list[dict[str, str]]:
        state = self._states.get(user_id, {})
        return list(state.get("history", []))

    def append_chat_history(self, user_id: str, query: str, response: str) -> None:
        history = self.get_chat_history(user_id)
        history.append({"user_message": query, "assistant_response": response})
        self.save_user_state(user_id, {"history": history[-5:]})

    def get_response_language(self, user_id: str) -> str | None:
        state = self._states.get(user_id, {})
        return state.get("response_language")

    def set_response_language(self, user_id: str, language: str) -> None:
        self.save_user_state(user_id, {"response_language": language})

    def clear_response_language(self, user_id: str) -> None:
        state = self._states.get(user_id, {"user_id": user_id}).copy()
        state.pop("response_language", None)
        self._states[user_id] = state

    def get_last_response_language(self, user_id: str) -> str | None:
        state = self._states.get(user_id, {})
        return state.get("last_response_language")

    def set_last_response_language(self, user_id: str, language: str) -> None:
        self.save_user_state(user_id, {"last_response_language": language})

    def get_agentic_strength(self, user_id: str) -> str | None:
        state = self._states.get(user_id, {})
        return state.get("agentic_strength")

    def set_agentic_strength(self, user_id: str, strength: str) -> None:
        normalized = strength.strip().lower()
        if normalized not in {"normal", "low", "very_low"}:
            raise ValueError(f"Invalid agentic strength: {strength}")
        self.save_user_state(user_id, {"agentic_strength": normalized})

    def get_dev_agentic_mcp(self, user_id: str) -> bool | None:
        state = self._states.get(user_id, {})
        value = state.get("dev_agentic_mcp")
        return bool(value) if value is not None else None

    def set_dev_agentic_mcp(self, user_id: str, enabled: bool) -> None:
        self.save_user_state(user_id, {"dev_agentic_mcp": bool(enabled)})

    def set_first_interaction(self, user_id: str, is_first: bool) -> None:
        self.save_user_state(user_id, {"first_interaction": is_first})

    def is_first_interaction(self, user_id: str) -> bool:
        state = self._states.get(user_id, {})
        return bool(state.get("first_interaction", True))


class NullChromaPort(ChromaPort):
    """Minimal Chroma port stub for tests that don't touch vector search."""

    def get_collection(self, name: str) -> Any | None:  # noqa: ANN401
        return None

    def list_collections(self) -> list[str]:
        return []

    def create_collection(self, name: str) -> None:  # noqa: D401
        del name

    def delete_collection(self, name: str) -> None:  # noqa: D401
        del name

    def delete_document(self, name: str, document_id: str) -> None:  # noqa: D401
        del name, document_id

    def count_documents(self, name: str) -> int:
        del name
        return 0

    def get_document_text_and_metadata(
        self, name: str, document_id: str
    ) -> tuple[str, Mapping[str, Any]]:
        del name, document_id
        return "", {}

    def list_document_ids(self, name: str) -> list[str]:
        del name
        return []

    def iter_batches(
        self,
        name: str,
        *,
        batch_size: int = 1000,
        include_embeddings: bool = False,
    ) -> Iterable[dict[str, Any]]:
        del name, batch_size, include_embeddings
        return []

    def get_collections_pair(self, source: str, dest: str) -> tuple[Any, Any]:
        del source, dest
        raise RuntimeError("Collection pair not supported in test stub.")

    def max_numeric_id(self, name: str) -> int:
        del name
        return 0


@pytest.fixture(autouse=True)
def service_container() -> Iterator[ServiceContainer]:
    """Provide a default in-memory service container for tests."""
    container = ServiceContainer(
        chroma=NullChromaPort(),
        user_state=InMemoryUserStatePort(),
        intent_router=IntentRouter(),
    )
    runtime.set_services(container)
    yield container
    runtime.clear_services()
