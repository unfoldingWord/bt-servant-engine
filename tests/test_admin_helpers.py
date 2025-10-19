"""Unit coverage for admin helper functions."""

from __future__ import annotations

from typing import cast
from unittest.mock import Mock

from bt_servant_engine.core.ports import ChromaPort
from bt_servant_engine.services.admin.datastore import (
    MergeRequest,
    MergeTaskRunner,
    MergeTaskStatus,
)
from bt_servant_engine.services.admin.merge_helpers import (
    MetadataTaggingConfig,
    apply_metadata_tags,
)

SOURCE_COLLECTION_COUNT = 5
DESTINATION_COLLECTION_COUNT = 2


def test_apply_metadata_tags_adds_fields() -> None:
    """Metadata tags stamp source, task, and optional original ids."""
    metadatas = [{"existing": 1}, {}]
    config = MetadataTaggingConfig(
        enabled=True,
        tag_key="_merged_from",
        source="src",
        task_id="task",
        tag_timestamp=False,
        source_ids=["a", "b"],
        source_id_key="_source_id",
    )
    tagged = apply_metadata_tags(metadatas, config)
    assert tagged is not None
    assert tagged[0]["_merged_from"] == "src"
    assert tagged[0]["_merge_task_id"] == "task"
    assert tagged[0]["_source_id"] == "a"


def test_merge_worker_dry_run() -> None:
    """Merge worker dry-run populates counts and finishes without modifying data."""
    src = Mock()
    dest = Mock()
    src.count.return_value = SOURCE_COLLECTION_COUNT
    dest.count.return_value = DESTINATION_COLLECTION_COUNT

    class _StubChromaPort:
        """Stubbed Chroma port exposing only the methods exercised in tests."""

        def __init__(self, source_handle, dest_handle) -> None:
            """Store handles to return for source and destination collections."""
            self._source = source_handle
            self._dest = dest_handle

        def get_collections_pair(self, source: str, dest: str):
            """Return the cached source and destination handles."""
            del source, dest
            return self._source, self._dest

        def max_numeric_id(self, name: str) -> int:
            """Return a constant max id for deterministic dry-run behavior."""
            del name
            return 0

    task = MergeTaskStatus(task_id="t", source="src", dest="dest", status="pending")
    request = MergeRequest(source="src", dry_run=True, on_duplicate="skip")

    runner = MergeTaskRunner(cast(ChromaPort, _StubChromaPort(src, dest)), task, request)
    runner.execute()

    assert task.status == "completed"
    assert task.total == SOURCE_COLLECTION_COUNT
