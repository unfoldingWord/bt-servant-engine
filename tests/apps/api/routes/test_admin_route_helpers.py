"""Unit tests for helper utilities in the admin routes module."""

from __future__ import annotations

import time
from types import SimpleNamespace
from typing import Generator

import pytest

from bt_servant_engine.apps.api.routes import admin
from bt_servant_engine.apps.api.routes import admin_merge_helpers

MIN_TOKEN_ESTIMATE = 1
TOKEN_HEURISTIC_INPUT_LENGTH = 20
TOKEN_HEURISTIC_DIVISOR = 4
EXPECTED_TOKEN_ESTIMATE = TOKEN_HEURISTIC_INPUT_LENGTH // TOKEN_HEURISTIC_DIVISOR
MAX_TOKEN_BUDGET = 10
DOC_TOKEN_COST = 2
DUPLICATE_PREVIEW_LIMIT = 5
MERGE_BATCH_SIZE = 2
SMALL_TOKEN_BUDGET = 4
IN_PROGRESS_DURATION_SECONDS = 2
LARGE_TOKEN_BUDGET = MAX_TOKEN_BUDGET * 10


def test_estimate_tokens_handles_empty_and_length() -> None:
    """Estimate tokens should fall back to minimum and scale with input length."""
    assert admin.estimate_tokens(None) == MIN_TOKEN_ESTIMATE
    assert admin.estimate_tokens("") == MIN_TOKEN_ESTIMATE
    # 20 chars -> 5 tokens using //4 heuristic
    assert admin.estimate_tokens("x" * TOKEN_HEURISTIC_INPUT_LENGTH) == EXPECTED_TOKEN_ESTIMATE


def test_yield_token_limited_slices_handles_edge_cases() -> None:
    """Slice helper splits inputs under the token budget and keeps metadata aligned."""
    assert not admin.yield_token_limited_slices([], None, None, max_tokens=LARGE_TOKEN_BUDGET)
    ids = ["a", "b", "c"]
    result = admin.yield_token_limited_slices(
        ids,
        None,
        [{"meta": 1}],
        max_tokens=MAX_TOKEN_BUDGET,
    )
    assert result == [(["a", "b", "c"], [], [{"meta": 1}])]

    docs = ["w" * 8, "x" * 8, "y" * 8]  # -> 2 tokens each
    metas = [{"id": 1}, {"id": 2}, {"id": 3}]
    batched = admin.yield_token_limited_slices(
        ids,
        docs,
        metas,
        max_tokens=SMALL_TOKEN_BUDGET,
    )
    # Expect batches of size two then one based on token budget
    assert batched == [
        (["a", "b"], ["w" * 8, "x" * 8], [{"id": 1}, {"id": 2}]),
        (["c"], ["y" * 8], [{"id": 3}]),
    ]


def test_apply_metadata_tags_includes_optional_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    """Metadata tagging stamps task markers and preserves the original dict."""
    monkeypatch.setattr(
        "bt_servant_engine.apps.api.routes.admin_merge_helpers.now_iso",
        lambda: "2025-01-01T00:00:00Z",
    )
    original = [{"existing": True}]
    config = admin.MetadataTaggingConfig(
        enabled=True,
        tag_key="_merged_from",
        source="source-col",
        task_id="task-123",
        tag_timestamp=True,
        source_ids=["src-1"],
        source_id_key="_source_id",
    )
    stamped = admin.apply_metadata_tags(original, config)
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

    assert admin.apply_metadata_tags(
        None,
        admin.MetadataTaggingConfig(
            enabled=True,
            tag_key="_merged_from",
            source="x",
            task_id="y",
            tag_timestamp=False,
        ),
    ) is None
    disabled = admin.apply_metadata_tags(
        original,
        admin.MetadataTaggingConfig(
            enabled=False,
            tag_key="_merged_from",
            source="source-col",
            task_id="task-123",
            tag_timestamp=False,
        ),
    )
    assert disabled is original


def test_compute_duplicate_preview_detects_duplicates(monkeypatch: pytest.MonkeyPatch) -> None:
    """Duplicate preview collects sample ids from overlapping batches."""

    def iter_batches(
        collection: SimpleNamespace, batch_size: int, include_embeddings: bool
    ) -> Generator[dict[str, list[str]], None, None]:
        del batch_size, include_embeddings
        for ids in collection.batches:
            yield {"ids": ids}

    monkeypatch.setattr(admin_merge_helpers, "iter_collection_batches", iter_batches)

    dest = SimpleNamespace(batches=[["1", "2", "3"]])
    source = SimpleNamespace(batches=[["0", "2"], ["4"]])

    found, preview = admin.compute_duplicate_preview(
        source, dest, limit=DUPLICATE_PREVIEW_LIMIT, batch_size=MERGE_BATCH_SIZE
    )
    assert found is True
    assert preview == ["2"]

    empty_source = SimpleNamespace(batches=[["10"], ["11"]])
    found, preview = admin.compute_duplicate_preview(
        empty_source, dest, limit=DUPLICATE_PREVIEW_LIMIT, batch_size=MERGE_BATCH_SIZE
    )
    assert found is False
    assert not preview


def test_update_eta_metrics_updates_progress_fields() -> None:
    """ETA metrics compute rate fields when a task progresses."""
    task = admin.MergeTaskStatus(
        task_id="task",
        source="src",
        dest="dest",
        status="running",
        total=MAX_TOKEN_BUDGET,
        completed=MAX_TOKEN_BUDGET // 2,
        started_at=time.time() - IN_PROGRESS_DURATION_SECONDS,
    )
    admin.update_eta_metrics(task)
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
    admin.update_eta_metrics(idle_task)
    assert idle_task.docs_per_second is None
    assert idle_task.eta_seconds is None
    assert idle_task.eta_at is None
