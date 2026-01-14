"""Tests for the user state TinyDB adapter."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, cast

import pytest
from tinydb import Query, TinyDB

from bt_servant_engine.adapters import user_state


@pytest.fixture(name="temp_user_db")
def _temp_user_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Provide a temporary TinyDB instance and patch the module globals."""
    db_path = tmp_path / "db.json"
    db = TinyDB(db_path)
    monkeypatch.setattr(user_state, "_db", db)
    monkeypatch.setattr(user_state, "DB_PATH", db_path)
    yield db
    db.close()


def test_chat_history_roundtrip_respects_max(
    temp_user_db: TinyDB, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Chat history stays capped and preserves the latest response."""
    del temp_user_db
    # Patch the config to use a small storage max for testing
    from bt_servant_engine.core import config as config_module

    monkeypatch.setattr(config_module.config, "CHAT_HISTORY_STORAGE_MAX", 5)

    user_id = "tester"
    for i in range(6):  # Add one more than the max
        user_state.update_user_chat_history(user_id, f"q{i}", f"r{i}")

    history = user_state.get_user_chat_history(user_id)
    assert len(history) == 5
    assert history[-1]["assistant_response"] == "r5"


def test_response_language_roundtrip(temp_user_db: TinyDB) -> None:
    """Response language round-trips through the adapter."""
    del temp_user_db
    user_id = "lang-user"
    assert user_state.get_user_response_language(user_id) is None
    user_state.set_user_response_language(user_id, "es")
    assert user_state.get_user_response_language(user_id) == "es"


def test_clear_response_language_removes_value(temp_user_db: TinyDB) -> None:
    """Clearing response language removes the persisted field."""
    del temp_user_db
    user_id = "clear-lang"
    user_state.set_user_response_language(user_id, "fr")
    assert user_state.get_user_response_language(user_id) == "fr"
    user_state.clear_user_response_language(user_id)
    assert user_state.get_user_response_language(user_id) is None


def test_last_response_language_roundtrip(temp_user_db: TinyDB) -> None:
    """Last response language fields round-trip via helper functions."""
    del temp_user_db
    user_id = "last-lang"
    assert user_state.get_user_last_response_language(user_id) is None
    user_state.set_user_last_response_language(user_id, "nl")
    assert user_state.get_user_last_response_language(user_id) == "nl"


def test_agentic_strength_roundtrip(temp_user_db: TinyDB) -> None:
    """Agentic strength enforces the allowed value set."""
    del temp_user_db
    user_id = "agentic"
    assert user_state.get_user_agentic_strength(user_id) is None

    user_state.set_user_agentic_strength(user_id, "LOW")
    assert user_state.get_user_agentic_strength(user_id) == "low"

    with pytest.raises(ValueError):
        user_state.set_user_agentic_strength(user_id, "invalid")

    # Corrupt stored data should fall back to None
    user_state.get_user_db().table("users").upsert(
        {"user_id": user_id, "agentic_strength": "unsupported"}, Query().user_id == user_id
    )
    assert user_state.get_user_agentic_strength(user_id) is None


def test_dev_agentic_mcp_roundtrip(temp_user_db: TinyDB) -> None:
    """Dev MCP toggle round-trips boolean values."""
    del temp_user_db
    user_id = "dev-mcp"
    assert user_state.get_user_dev_agentic_mcp(user_id) is None

    user_state.set_user_dev_agentic_mcp(user_id, True)
    assert user_state.get_user_dev_agentic_mcp(user_id) is True

    user_state.set_user_dev_agentic_mcp(user_id, False)
    assert user_state.get_user_dev_agentic_mcp(user_id) is False


def test_first_interaction_flags(temp_user_db: TinyDB) -> None:
    """First-interaction flag flips and persists via adapter helpers."""
    del temp_user_db
    user_id = "first"
    assert user_state.is_first_interaction(user_id) is True

    user_state.set_first_interaction(user_id, False)
    assert user_state.is_first_interaction(user_id) is False

    user_state.set_first_interaction(user_id, True)
    assert user_state.is_first_interaction(user_id) is True


# pylint: disable=too-many-locals
def test_user_state_adapter_methods_delegate(  # noqa: PLR0915
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Adapter delegates work to the module-level helpers."""
    adapter = user_state.UserStateAdapter()

    calls: list[str] = []

    def record(name: str) -> None:
        calls.append(name)

    def fake_get_history(uid: str) -> list[dict[str, Any]]:
        record(f"get_history:{uid}")
        return []

    def fake_get_history_for_llm(uid: str) -> list[dict[str, str]]:
        record(f"get_history_for_llm:{uid}")
        return []

    def fake_update_history(
        uid: str, q: str, r: str, created_at: datetime | None = None
    ) -> None:
        record(f"append:{uid}:{q}:{r}")

    def fake_get_lang(uid: str) -> str | None:
        record(f"get_lang:{uid}")
        return cast(str | None, None)

    def fake_set_lang(uid: str, lang: str) -> None:
        record(f"set_lang:{uid}:{lang}")

    def fake_clear_lang(uid: str) -> None:
        record(f"clear_lang:{uid}")

    def fake_get_last_lang(uid: str) -> str | None:
        record(f"get_last_lang:{uid}")
        return cast(str | None, None)

    def fake_set_last_lang(uid: str, lang: str) -> None:
        record(f"set_last_lang:{uid}:{lang}")

    def fake_get_strength(uid: str) -> str | None:
        record(f"get_strength:{uid}")
        return cast(str | None, None)

    def fake_set_strength(uid: str, strength: str) -> None:
        record(f"set_strength:{uid}:{strength}")

    def fake_get_dev_mcp(uid: str) -> bool | None:
        record(f"get_dev:{uid}")
        return cast(bool | None, None)

    def fake_set_dev_mcp(uid: str, enabled: bool) -> None:
        record(f"set_dev:{uid}:{enabled}")

    def fake_set_first(uid: str, val: bool) -> None:
        record(f"set_first:{uid}:{val}")

    def fake_is_first(uid: str) -> bool:
        record(f"is_first:{uid}")
        return True

    monkeypatch.setattr(user_state, "get_user_chat_history", fake_get_history)
    monkeypatch.setattr(user_state, "get_user_chat_history_for_llm", fake_get_history_for_llm)
    monkeypatch.setattr(user_state, "update_user_chat_history", fake_update_history)
    monkeypatch.setattr(user_state, "get_user_response_language", fake_get_lang)
    monkeypatch.setattr(user_state, "set_user_response_language", fake_set_lang)
    monkeypatch.setattr(user_state, "clear_user_response_language", fake_clear_lang)
    monkeypatch.setattr(user_state, "get_user_last_response_language", fake_get_last_lang)
    monkeypatch.setattr(user_state, "set_user_last_response_language", fake_set_last_lang)
    monkeypatch.setattr(user_state, "get_user_agentic_strength", fake_get_strength)
    monkeypatch.setattr(user_state, "set_user_agentic_strength", fake_set_strength)
    monkeypatch.setattr(user_state, "get_user_dev_agentic_mcp", fake_get_dev_mcp)
    monkeypatch.setattr(user_state, "set_user_dev_agentic_mcp", fake_set_dev_mcp)
    monkeypatch.setattr(user_state, "set_first_interaction", fake_set_first)
    monkeypatch.setattr(user_state, "is_first_interaction", fake_is_first)

    adapter.get_chat_history("u1")
    adapter.get_chat_history_for_llm("u1")
    adapter.append_chat_history("u1", "hi", "hello")
    adapter.get_response_language("u1")
    adapter.set_response_language("u1", "fr")
    adapter.clear_response_language("u1")
    adapter.get_last_response_language("u1")
    adapter.set_last_response_language("u1", "nl")
    adapter.get_agentic_strength("u1")
    adapter.set_agentic_strength("u1", "normal")
    adapter.get_dev_agentic_mcp("u1")
    adapter.set_dev_agentic_mcp("u1", True)
    adapter.set_first_interaction("u1", False)
    adapter.is_first_interaction("u1")

    assert calls == [
        "get_history:u1",
        "get_history_for_llm:u1",
        "append:u1:hi:hello",
        "get_lang:u1",
        "set_lang:u1:fr",
        "clear_lang:u1",
        "get_last_lang:u1",
        "set_last_lang:u1:nl",
        "get_strength:u1",
        "set_strength:u1:normal",
        "get_dev:u1",
        "set_dev:u1:True",
        "set_first:u1:False",
        "is_first:u1",
    ]
