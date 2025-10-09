"""Unit tests for helper utilities in the admin routes module."""

from __future__ import annotations

import time

import pytest

from bt_servant_engine.apps.api.routes import admin

# pylint: disable=missing-function-docstring,missing-class-docstring,too-few-public-methods,protected-access


def test_estimate_tokens_handles_empty_and_length() -> None:
    assert admin._estimate_tokens(None) == 1  # pylint: disable=protected-access
    assert admin._estimate_tokens("") == 1  # pylint: disable=protected-access
    # 20 chars -> 5 tokens using //4 heuristic
    assert admin._estimate_tokens("x" * 20) == 5  # pylint: disable=protected-access


def test_yield_token_limited_slices_handles_edge_cases() -> None:
    assert not admin._yield_token_limited_slices([], None, None, max_tokens=100)  # pylint: disable=protected-access
    ids = ["a", "b", "c"]
    result = admin._yield_token_limited_slices(  # pylint: disable=protected-access
        ids,
        None,
        [{"meta": 1}],
        max_tokens=10,
    )
    assert result == [(["a", "b", "c"], [], [{"meta": 1}])]

    docs = ["w" * 8, "x" * 8, "y" * 8]  # -> 2 tokens each
    metas = [{"id": 1}, {"id": 2}, {"id": 3}]
    batched = admin._yield_token_limited_slices(  # pylint: disable=protected-access
        ids,
        docs,
        metas,
        max_tokens=4,
    )
    # Expect batches of size two then one based on token budget
    assert batched == [
        (["a", "b"], ["w" * 8, "x" * 8], [{"id": 1}, {"id": 2}]),
        (["c"], ["y" * 8], [{"id": 3}]),
    ]


def test_apply_metadata_tags_includes_optional_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(admin, "_now_iso", lambda: "2025-01-01T00:00:00Z")  # pylint: disable=protected-access
    original = [{"existing": True}]
    stamped = admin._apply_metadata_tags(  # pylint: disable=protected-access
        original,
        enabled=True,
        tag_key="_merged_from",
        source="source-col",
        task_id="task-123",
        tag_timestamp=True,
        source_ids=["src-1"],
        source_id_key="_source_id",
    )
    assert stamped is not original
    assert stamped == [
        {
            "existing": True,
            "_merged_from": "source-col",
            "_merge_task_id": "task-123",
            "_merged_at": "2025-01-01T00:00:00Z",
            "_source_id": "src-1",
        }
    ]

    assert admin._apply_metadata_tags(None, enabled=True, tag_key="_merged_from", source="x", task_id="y", tag_timestamp=False) is None  # pylint: disable=protected-access
    disabled = admin._apply_metadata_tags(  # pylint: disable=protected-access
        original,
        enabled=False,
        tag_key="_merged_from",
        source="source-col",
        task_id="task-123",
        tag_timestamp=False,
    )
    assert disabled is original


def test_compute_duplicate_preview_detects_duplicates(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyCollection:
        def __init__(self, batches: list[list[str]]) -> None:
            self._batches = batches

    def iter_batches(collection: DummyCollection, batch_size: int, include_embeddings: bool):  # noqa: D401
        del batch_size, include_embeddings
        for ids in collection._batches:
            yield {"ids": ids}

    monkeypatch.setattr(admin, "iter_collection_batches", iter_batches)

    dest = DummyCollection([["1", "2", "3"]])
    source = DummyCollection([["0", "2"], ["4"]])

    found, preview = admin._compute_duplicate_preview(  # pylint: disable=protected-access
        source, dest, limit=5, batch_size=2
    )
    assert found is True
    assert preview == ["2"]

    empty_source = DummyCollection([["10"], ["11"]])
    found, preview = admin._compute_duplicate_preview(  # pylint: disable=protected-access
        empty_source, dest, limit=5, batch_size=2
    )
    assert found is False
    assert not preview


def test_update_eta_metrics_updates_progress_fields() -> None:
    task = admin.MergeTaskStatus(
        task_id="task",
        source="src",
        dest="dest",
        status="running",
        total=10,
        completed=5,
        started_at=time.time() - 2,
    )
    admin._update_eta_metrics(task)  # pylint: disable=protected-access
    assert task.docs_per_second is not None
    assert task.eta_seconds is not None
    assert task.eta_at is not None

    idle_task = admin.MergeTaskStatus(
        task_id="idle",
        source="s",
        dest="d",
        status="pending",
        total=0,
        completed=0,
        started_at=None,
    )
    admin._update_eta_metrics(idle_task)  # pylint: disable=protected-access
    assert idle_task.docs_per_second is None
    assert idle_task.eta_seconds is None
    assert idle_task.eta_at is None
