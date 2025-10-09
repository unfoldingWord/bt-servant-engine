"""Tests for log directory resolution logic."""

from __future__ import annotations

import pathlib
from types import SimpleNamespace

import pytest

from bt_servant_engine.core import logging as core_logging

# pylint: disable=missing-function-docstring


def _make_settings(log_dir: pathlib.Path | None, data_dir: pathlib.Path) -> SimpleNamespace:
    return SimpleNamespace(
        BT_SERVANT_LOG_LEVEL="info",
        BT_SERVANT_LOG_DIR=log_dir,
        DATA_DIR=data_dir,
    )


def test_resolve_logs_dir_prefers_override(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    override = tmp_path / "custom-logs"
    data_dir = tmp_path / "data"
    settings = _make_settings(override, data_dir)
    monkeypatch.setattr(core_logging, "settings", settings)
    monkeypatch.setattr(core_logging, "ROOT_DIR", tmp_path / "app")
    monkeypatch.setattr(core_logging, "BASE_DIR", tmp_path / "app" / "pkg")

    resolved = core_logging._resolve_logs_dir()  # pylint: disable=protected-access
    assert resolved == override
    assert resolved.exists()


def test_resolve_logs_dir_falls_back_to_root_logs(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root_dir = tmp_path / "deploy"
    base_dir = root_dir / "pkg"
    data_dir = tmp_path / "data"
    settings = _make_settings(None, data_dir)

    monkeypatch.setattr(core_logging, "settings", settings)
    monkeypatch.setattr(core_logging, "ROOT_DIR", root_dir)
    monkeypatch.setattr(core_logging, "BASE_DIR", base_dir)

    fallback = core_logging._resolve_logs_dir()  # pylint: disable=protected-access
    assert fallback == root_dir / "logs"
    assert fallback.exists()


def test_resolve_logs_dir_skips_unwritable_paths(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    override = tmp_path / "override"
    root_dir = tmp_path / "root"
    base_dir = root_dir / "pkg"
    data_dir = tmp_path / "data"
    settings = _make_settings(override, data_dir)

    monkeypatch.setattr(core_logging, "settings", settings)
    monkeypatch.setattr(core_logging, "ROOT_DIR", root_dir)
    monkeypatch.setattr(core_logging, "BASE_DIR", base_dir)

    original_mkdir = pathlib.Path.mkdir

    def guarded_mkdir(path_obj: pathlib.Path, *args, **kwargs):
        if path_obj == override:
            raise PermissionError("unwritable")
        return original_mkdir(path_obj, *args, **kwargs)

    monkeypatch.setattr(pathlib.Path, "mkdir", guarded_mkdir)

    resolved = core_logging._resolve_logs_dir()  # pylint: disable=protected-access
    assert resolved == root_dir / "logs"
    assert resolved.exists()
