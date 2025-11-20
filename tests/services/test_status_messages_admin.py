"""Tests for status message cache CRUD helpers."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from bt_servant_engine.services import status_messages


def _setup_temp_store(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    data = {"TEST_KEY": {"en": "English source"}}
    path = tmp_path / "status_messages_data.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    store = status_messages.StatusMessageStore(
        status_messages=copy.deepcopy(data),
        dynamic_cache={},
    )
    monkeypatch.setattr(status_messages, "_STATUS_MESSAGES_PATH", path)
    monkeypatch.setattr(status_messages, "_STATUS_STORE", store)
    return path


def test_set_translation_updates_cache_and_disk(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Setting a translation writes to disk, cache, and in-memory map."""
    path = _setup_temp_store(monkeypatch, tmp_path)

    status_messages.set_status_message_translation("TEST_KEY", "am", "Amharic text")

    translations = status_messages.get_status_message_translations("TEST_KEY")
    assert translations["am"] == "Amharic text"
    on_disk = json.loads(path.read_text(encoding="utf-8"))
    assert on_disk["TEST_KEY"]["am"] == "Amharic text"
    assert status_messages._STATUS_STORE.dynamic_cache[("TEST_KEY", "am")] == "Amharic text"  # noqa: SLF001


def test_delete_translation_removes_cache_and_disk(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Deleting a translation removes it from disk and cache."""
    path = _setup_temp_store(monkeypatch, tmp_path)
    status_messages.set_status_message_translation("TEST_KEY", "am", "Amharic text")
    status_messages.delete_status_message_translation("TEST_KEY", "am")

    translations = status_messages.get_status_message_translations("TEST_KEY")
    assert "am" not in translations
    on_disk = json.loads(path.read_text(encoding="utf-8"))
    assert "am" not in on_disk["TEST_KEY"]
    assert ("TEST_KEY", "am") not in status_messages._STATUS_STORE.dynamic_cache  # noqa: SLF001


def test_guard_english_overrides(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """English source translations cannot be overridden or deleted."""
    _setup_temp_store(monkeypatch, tmp_path)

    with pytest.raises(ValueError):
        status_messages.set_status_message_translation("TEST_KEY", "en", "nope")
    with pytest.raises(ValueError):
        status_messages.delete_status_message_translation("TEST_KEY", "en")
    with pytest.raises(KeyError):
        status_messages.set_status_message_translation("UNKNOWN", "am", "text")
