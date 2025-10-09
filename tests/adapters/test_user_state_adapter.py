"""Tests for the user state TinyDB adapter."""

# pylint: disable=missing-function-docstring,missing-class-docstring,redefined-outer-name,useless-return

from __future__ import annotations

from pathlib import Path

import pytest
from tinydb import Query, TinyDB

from bt_servant_engine.adapters import user_state


@pytest.fixture()
def temp_user_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Provide a temporary TinyDB instance and patch the module globals."""
    db_path = tmp_path / "db.json"
    db = TinyDB(db_path)
    monkeypatch.setattr(user_state, "_db", db)
    monkeypatch.setattr(user_state, "DB_PATH", db_path)
    yield db
    db.close()


def test_chat_history_roundtrip_respects_max(temp_user_db: TinyDB) -> None:
    user_id = "tester"
    for i in range(user_state.CHAT_HISTORY_MAX + 1):
        user_state.update_user_chat_history(user_id, f"q{i}", f"r{i}")

    history = user_state.get_user_chat_history(user_id)
    assert len(history) == user_state.CHAT_HISTORY_MAX
    assert history[-1]["assistant_response"] == "r5"


def test_response_language_roundtrip(temp_user_db: TinyDB) -> None:
    user_id = "lang-user"
    assert user_state.get_user_response_language(user_id) is None
    user_state.set_user_response_language(user_id, "es")
    assert user_state.get_user_response_language(user_id) == "es"


def test_agentic_strength_roundtrip(temp_user_db: TinyDB) -> None:
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


def test_first_interaction_flags(temp_user_db: TinyDB) -> None:
    user_id = "first"
    assert user_state.is_first_interaction(user_id) is True

    user_state.set_first_interaction(user_id, False)
    assert user_state.is_first_interaction(user_id) is False

    user_state.set_first_interaction(user_id, True)
    assert user_state.is_first_interaction(user_id) is True


def test_user_state_adapter_methods_delegate(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = user_state.UserStateAdapter()

    calls: list[str] = []

    def record(name: str) -> None:
        calls.append(name)

    def fake_get_history(uid: str) -> list[dict[str, str]]:
        record(f"get_history:{uid}")
        return []

    def fake_update_history(uid: str, q: str, r: str) -> None:
        record(f"append:{uid}:{q}:{r}")

    def fake_get_lang(uid: str) -> str | None:
        record(f"get_lang:{uid}")
        return None

    def fake_set_lang(uid: str, lang: str) -> None:
        record(f"set_lang:{uid}:{lang}")

    def fake_get_strength(uid: str) -> str | None:
        record(f"get_strength:{uid}")
        return None

    def fake_set_strength(uid: str, strength: str) -> None:
        record(f"set_strength:{uid}:{strength}")

    def fake_set_first(uid: str, val: bool) -> None:
        record(f"set_first:{uid}:{val}")

    def fake_is_first(uid: str) -> bool:
        record(f"is_first:{uid}")
        return True

    monkeypatch.setattr(user_state, "get_user_chat_history", fake_get_history)
    monkeypatch.setattr(user_state, "update_user_chat_history", fake_update_history)
    monkeypatch.setattr(user_state, "get_user_response_language", fake_get_lang)
    monkeypatch.setattr(user_state, "set_user_response_language", fake_set_lang)
    monkeypatch.setattr(user_state, "get_user_agentic_strength", fake_get_strength)
    monkeypatch.setattr(user_state, "set_user_agentic_strength", fake_set_strength)
    monkeypatch.setattr(user_state, "set_first_interaction", fake_set_first)
    monkeypatch.setattr(user_state, "is_first_interaction", fake_is_first)

    adapter.get_chat_history("u1")
    adapter.append_chat_history("u1", "hi", "hello")
    adapter.get_response_language("u1")
    adapter.set_response_language("u1", "fr")
    adapter.get_agentic_strength("u1")
    adapter.set_agentic_strength("u1", "normal")
    adapter.set_first_interaction("u1", False)
    adapter.is_first_interaction("u1")

    assert calls == [
        "get_history:u1",
        "append:u1:hi:hello",
        "get_lang:u1",
        "set_lang:u1:fr",
        "get_strength:u1",
        "set_strength:u1:normal",
        "set_first:u1:False",
        "is_first:u1",
    ]
