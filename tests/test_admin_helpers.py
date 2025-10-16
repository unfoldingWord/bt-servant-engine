"""Unit coverage for admin helper functions."""

from __future__ import annotations

from unittest.mock import Mock

from bt_servant_engine.apps.api.routes import admin

SOURCE_COLLECTION_COUNT = 5
DESTINATION_COLLECTION_COUNT = 2


def test_apply_metadata_tags_adds_fields() -> None:
    """Metadata tags stamp source, task, and optional original ids."""
    metadatas = [{"existing": 1}, {}]
    config = admin.MetadataTaggingConfig(
        enabled=True,
        tag_key="_merged_from",
        source="src",
        task_id="task",
        tag_timestamp=False,
        source_ids=["a", "b"],
        source_id_key="_source_id",
    )
    tagged = admin.apply_metadata_tags(metadatas, config)
    assert tagged is not None
    assert tagged[0]["_merged_from"] == "src"
    assert tagged[0]["_merge_task_id"] == "task"
    assert tagged[0]["_source_id"] == "a"


def test_merge_worker_dry_run(monkeypatch) -> None:
    """Merge worker dry-run populates counts and finishes without modifying data."""
    src = Mock()
    dest = Mock()
    src.count.return_value = SOURCE_COLLECTION_COUNT
    dest.count.return_value = DESTINATION_COLLECTION_COUNT

    monkeypatch.setattr(admin, "get_chroma_collections_pair", lambda *_: (src, dest), raising=True)

    task = admin.MergeTaskStatus(task_id="t", source="src", dest="dest", status="pending")
    request = admin.MergeRequest(source="src", dry_run=True, on_duplicate="skip")

    admin.merge_worker(task, request)

    assert task.status == "completed"
    assert task.total == SOURCE_COLLECTION_COUNT
