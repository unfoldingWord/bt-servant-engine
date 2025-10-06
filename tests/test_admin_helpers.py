"""Unit coverage for admin helper functions."""
# pylint: disable=missing-function-docstring,protected-access,too-few-public-methods

from __future__ import annotations

from bt_servant_engine.apps.api.routes import admin


def test_apply_metadata_tags_adds_fields():
    metadatas = [{"existing": 1}, {}]
    tagged = admin._apply_metadata_tags(  # type: ignore[attr-defined]
        metadatas,
        enabled=True,
        tag_key="_merged_from",
        source="src",
        task_id="task",
        tag_timestamp=False,
        source_ids=["a", "b"],
        source_id_key="_source_id",
    )
    assert tagged is not None
    assert tagged[0]["_merged_from"] == "src"
    assert tagged[0]["_merge_task_id"] == "task"
    assert tagged[0]["_source_id"] == "a"


class _StubCollection:
    def __init__(self, count: int) -> None:
        self._count = count

    def count(self) -> int:
        return self._count


def test_merge_worker_dry_run(monkeypatch):
    src = _StubCollection(5)
    dest = _StubCollection(2)

    monkeypatch.setattr(admin, "get_chroma_collections_pair", lambda *_: (src, dest), raising=True)

    task = admin.MergeTaskStatus(task_id="t", source="src", dest="dest", status="pending")
    request = admin.MergeRequest(source="src", dry_run=True, on_duplicate="skip")

    admin._merge_worker(task, request)  # type: ignore[attr-defined]

    assert task.status == "completed"
    assert task.total == 5
