"""Service layer orchestrating administrative datastore operations."""

from __future__ import annotations

import asyncio
import concurrent.futures
import os
import sys
import threading
import time
import uuid
from collections import defaultdict
from typing import Any, DefaultDict, Mapping, MutableMapping, cast

from pydantic import BaseModel

from bt_servant_engine.core.exceptions import CollectionNotFoundError
from bt_servant_engine.core.logging import get_logger
from bt_servant_engine.core.ports import ChromaPort
from bt_servant_engine.services.cache_manager import cache_manager
from .merge_helpers import (
    MetadataTaggingConfig,
    apply_metadata_tags,
    compute_duplicate_preview,
    iter_collection_batches,
    yield_token_limited_slices,
)

logger = get_logger(__name__)

_EMBEDDING_MAX_TOKENS_PER_REQUEST = int(
    os.environ.get("OPENAI_EMBED_MAX_TOKENS_PER_REQUEST", "290000")
)


class Document(BaseModel):
    """Payload schema for adding/upserting a document to Chroma."""

    document_id: str
    collection: str
    name: str
    text: str
    metadata: dict[str, Any]


class CollectionCreate(BaseModel):
    """Payload schema for creating a Chroma collection."""

    name: str


class MergeRequest(BaseModel):
    """Payload schema for merging one Chroma collection into another."""

    source: str
    mode: str = "copy"  # "copy" | "move"
    on_duplicate: str = "fail"  # "fail" | "skip" | "overwrite"
    create_new_id: bool = False
    batch_size: int = 1000
    use_source_embeddings: bool = False  # default re-embed
    dry_run: bool = False
    duplicates_preview_limit: int = 100
    tag_metadata: bool = True
    tag_metadata_key: str = "_merged_from"
    tag_metadata_timestamp: bool = True
    # When enabled, stamp the original source id on each inserted doc
    tag_source_id: bool = True
    tag_source_id_key: str = "_source_id"
    sleep_between_batches_ms: int = 0


class MergeTaskStatus(BaseModel):
    """Status model for a merge task."""

    task_id: str
    source: str
    dest: str
    status: str  # pending | running | completed | failed | cancelled
    total: int = 0
    completed: int = 0
    skipped: int = 0
    overwritten: int = 0
    deleted_from_source: int = 0
    duplicates_found: bool | None = None
    duplicate_preview: list[str] | None = None
    error: str | None = None
    started_at: float | None = None
    finished_at: float | None = None
    next_id_start: int | None = None
    docs_per_second: float | None = None
    eta_seconds: float | None = None
    eta_at: float | None = None


def _update_eta_metrics(task: MergeTaskStatus) -> None:
    """Update docs/sec, remaining seconds, and ETA timestamp based on progress."""
    if not task.started_at:
        return
    now = time.time()
    elapsed = max(0.0, now - task.started_at)
    if elapsed <= 0 or task.completed <= 0:
        task.docs_per_second = None
        task.eta_seconds = None
        task.eta_at = None
        return
    rate = task.completed / elapsed
    task.docs_per_second = rate
    if task.total and task.total > task.completed and rate > 0:
        remaining = task.total - task.completed
        eta_sec = remaining / rate
        task.eta_seconds = eta_sec
        task.eta_at = now + eta_sec
    else:
        task.eta_seconds = 0.0
        task.eta_at = now


merge_global_semaphore = asyncio.Semaphore(1)
merge_collection_locks: DefaultDict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
_merge_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
_merge_tasks: dict[str, MergeTaskStatus] = {}
_merge_task_cancel_flags: dict[str, threading.Event] = {}


class MergeTaskRunner:
    """Orchestrate a merge operation while keeping helper methods small."""

    def __init__(self, chroma: ChromaPort, task: MergeTaskStatus, req: MergeRequest) -> None:
        self.chroma = chroma
        self.task = task
        self.req = req
        self.source_col: Any | None = None
        self.dest_col: Any | None = None
        self.next_id: int | None = None

    def execute(self) -> None:
        """Run the merge and record failure details if an exception is raised."""
        try:
            self.run()
        finally:
            exc = sys.exc_info()[1]
            if exc is not None:
                self._record_failure(exc)

    def run(self) -> None:
        """Perform the merge workflow after initial validation succeeds."""
        self._log_start()
        self.task.started_at = time.time()
        self.source_col, self.dest_col = self.chroma.get_collections_pair(
            self.req.source, self.task.dest
        )
        self._ensure_distinct_collections()
        self.task.total = self._count_source_documents()
        if self.req.dry_run:
            self._perform_dry_run()
            return
        if self._should_fail_on_duplicate():
            self._enforce_no_duplicates()
        self._initialize_id_allocator()
        for batch in iter_collection_batches(
            self.source_col,
            batch_size=self.req.batch_size,
            include_embeddings=self.req.use_source_embeddings,
        ):
            if self._is_cancelled():
                self._mark_cancelled()
                return
            self._process_batch(batch)
            if self._sleep_if_needed():
                return
        self._mark_completed()

    def _log_start(self) -> None:
        logger.info(
            (
                "merge start: task_id=%s source=%s dest=%s mode=%s create_new_id=%s "
                "on_duplicate=%s use_source_embeddings=%s batch_size=%d dry_run=%s"
            ),
            self.task.task_id,
            self.req.source,
            self.task.dest,
            self.req.mode,
            self.req.create_new_id,
            self.req.on_duplicate,
            self.req.use_source_embeddings,
            self.req.batch_size,
            self.req.dry_run,
        )

    def _ensure_distinct_collections(self) -> None:
        if self.req.source.strip() == self.task.dest.strip():
            raise ValueError("source and dest must be different collections")

    def _count_source_documents(self) -> int:
        if self.source_col is None:
            raise RuntimeError("source collection not initialized")
        try:
            source_count = self.source_col.count()
        except TypeError:
            source_count = self.source_col.count
        self._touch_destination_count()
        return int(source_count)

    def _touch_destination_count(self) -> None:
        if self.dest_col is None:
            raise RuntimeError("destination collection not initialized")
        try:
            self.dest_col.count()
        except TypeError:
            _ = self.dest_col.count

    def _perform_dry_run(self) -> None:
        if self.source_col is None or self.dest_col is None:
            raise RuntimeError("collections must be initialized before dry run")
        if not self.req.create_new_id and self.req.on_duplicate == "fail":
            dup_found, preview = compute_duplicate_preview(
                self.source_col,
                self.dest_col,
                limit=self.req.duplicates_preview_limit,
                batch_size=self.req.batch_size,
            )
            self.task.duplicates_found = dup_found
            self.task.duplicate_preview = preview
        if self.req.create_new_id:
            start_id = self.chroma.max_numeric_id(self.task.dest) + 1
            self.task.next_id_start = start_id
        self.task.status = "completed"
        self.task.finished_at = time.time()
        _update_eta_metrics(self.task)

    def _should_fail_on_duplicate(self) -> bool:
        return not self.req.create_new_id and self.req.on_duplicate == "fail"

    def _enforce_no_duplicates(self) -> None:
        if self.source_col is None or self.dest_col is None:
            raise RuntimeError("collections must be initialized before enforcing duplicates")
        dup_found, _ = compute_duplicate_preview(
            self.source_col,
            self.dest_col,
            limit=1,
            batch_size=self.req.batch_size,
        )
        if dup_found:
            raise ValueError("Duplicate ids detected; aborting due to on_duplicate=fail")

    def _initialize_id_allocator(self) -> None:
        if self.req.create_new_id:
            self.next_id = self.chroma.max_numeric_id(self.task.dest) + 1

    def _is_cancelled(self) -> bool:
        cancel_event = _merge_task_cancel_flags.get(self.task.task_id)
        return bool(cancel_event and cancel_event.is_set())

    def _mark_cancelled(self) -> None:
        self.task.status = "cancelled"
        self.task.finished_at = time.time()
        _update_eta_metrics(self.task)
        logger.info(
            "merge cancelled: task_id=%s source=%s dest=%s completed=%d total=%d",
            self.task.task_id,
            self.req.source,
            self.task.dest,
            self.task.completed,
            self.task.total,
        )

    def _process_batch(self, batch: dict[str, Any]) -> None:
        src_ids = [str(i) for i in (batch.get("ids") or [])]
        if not src_ids:
            return
        documents = cast(list[str] | None, batch.get("documents"))
        metadatas = cast(list[dict[str, Any]] | None, batch.get("metadatas"))
        embeddings = self._extract_embeddings(batch)
        dest_ids = self._determine_destination_ids(len(src_ids), src_ids)
        indexes = self._filter_duplicate_indices(dest_ids)
        if not indexes:
            return
        add_docs = self._slice_optional(documents, indexes)
        add_metas = self._slice_optional(metadatas, indexes)
        add_src_ids = [src_ids[i] for i in indexes]
        config = MetadataTaggingConfig(
            enabled=self.req.tag_metadata,
            tag_key=self.req.tag_metadata_key,
            source=self.req.source,
            task_id=self.task.task_id,
            tag_timestamp=self.req.tag_metadata_timestamp,
            source_ids=add_src_ids if self.req.tag_source_id else None,
            source_id_key=self.req.tag_source_id_key,
        )
        add_metas = apply_metadata_tags(add_metas, config)
        add_embs = self._slice_optional(embeddings, indexes)
        add_ids = [dest_ids[i] for i in indexes]
        self._insert_documents(add_ids, add_docs, add_metas, add_embs)
        self._log_insertions(src_ids, dest_ids, indexes)
        self._delete_source_if_needed(src_ids)

    def _extract_embeddings(self, batch: dict[str, Any]) -> list[list[float]] | None:
        if not self.req.use_source_embeddings:
            return None
        return cast(list[list[float]] | None, batch.get("embeddings"))

    def _determine_destination_ids(self, count: int, source_ids: list[str]) -> list[str]:
        if self.req.create_new_id:
            if self.next_id is None:
                raise RuntimeError("ID allocator not initialized")
            dest_ids = [str(i) for i in range(self.next_id, self.next_id + count)]
            self.next_id += count
            return dest_ids
        return list(source_ids)

    def _filter_duplicate_indices(self, dest_ids: list[str]) -> list[int]:
        if self.req.create_new_id or self.req.on_duplicate not in {"skip", "overwrite"}:
            return list(range(len(dest_ids)))
        if self.dest_col is None:
            raise RuntimeError("destination collection not initialized")
        existing = self.dest_col.get(ids=dest_ids)
        existing_ids = {str(i) for i in (existing.get("ids") or [])}
        if self.req.on_duplicate == "skip":
            if not existing_ids:
                return list(range(len(dest_ids)))
            self.task.skipped += len(existing_ids)
            return [i for i, did in enumerate(dest_ids) if did not in existing_ids]
        if existing_ids:
            self.dest_col.delete(ids=list(existing_ids))
            self.task.overwritten += len(existing_ids)
        return list(range(len(dest_ids)))

    def _slice_optional(self, values: list[Any] | None, indexes: list[int]) -> list[Any] | None:
        if values is None:
            return None
        return [values[i] for i in indexes]

    def _insert_documents(
        self,
        ids: list[str],
        documents: list[str] | None,
        metadatas: list[dict[str, Any]] | None,
        embeddings: list[list[float]] | None,
    ) -> None:
        if self.dest_col is None:
            raise RuntimeError("destination collection not initialized")
        if embeddings is not None:
            self.dest_col.add(
                ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings
            )
            self.task.completed += len(ids)
            _update_eta_metrics(self.task)
            return
        doc_list = documents or []
        slices = yield_token_limited_slices(
            ids,
            doc_list,
            metadatas,
            max_tokens=_EMBEDDING_MAX_TOKENS_PER_REQUEST,
        )
        if not slices:
            slices = [(ids, doc_list, metadatas)]
        if len(slices) > 1:
            logger.info(
                "split add into %d sub-batches (max_tokens=%d)",
                len(slices),
                _EMBEDDING_MAX_TOKENS_PER_REQUEST,
            )
        for sb_ids, sb_docs, sb_metas in slices:
            self.dest_col.add(ids=sb_ids, documents=sb_docs, metadatas=sb_metas)
            self.task.completed += len(sb_ids)
            _update_eta_metrics(self.task)

    def _log_insertions(self, src_ids: list[str], dest_ids: list[str], indexes: list[int]) -> None:
        for idx in indexes:
            logger.info(
                "doc_id %s from %s inserted into %s with id %s",
                src_ids[idx],
                self.req.source,
                self.task.dest,
                dest_ids[idx],
            )

    def _delete_source_if_needed(self, src_ids: list[str]) -> None:
        if self.req.mode != "move":
            return
        if self.source_col is None:
            raise RuntimeError("source collection not initialized")
        self.source_col.delete(ids=src_ids)
        self.task.deleted_from_source += len(src_ids)

    def _sleep_if_needed(self) -> bool:
        if self.req.sleep_between_batches_ms <= 0:
            return False
        if self._is_cancelled():
            self._mark_cancelled()
            return True
        delay = max(0.0, self.req.sleep_between_batches_ms / 1000.0)
        time.sleep(delay)
        if self._is_cancelled():
            self._mark_cancelled()
            return True
        return False

    def _mark_completed(self) -> None:
        self.task.status = "completed"
        self.task.finished_at = time.time()
        _update_eta_metrics(self.task)
        logger.info(
            (
                "merge completed: task_id=%s source=%s dest=%s completed=%d "
                "skipped=%d overwritten=%d deleted_from_source=%d total=%d"
            ),
            self.task.task_id,
            self.req.source,
            self.task.dest,
            self.task.completed,
            self.task.skipped,
            self.task.overwritten,
            self.task.deleted_from_source,
            self.task.total,
        )

    def _record_failure(self, exc: BaseException) -> None:
        if self.task.status in {"completed", "cancelled"}:
            return
        self.task.status = "failed"
        self.task.error = str(exc)
        self.task.finished_at = time.time()
        _update_eta_metrics(self.task)
        logger.error(
            ("merge failed: task_id=%s source=%s dest=%s error=%s status=%s completed=%d total=%d"),
            self.task.task_id,
            self.req.source,
            self.task.dest,
            self.task.error,
            self.task.status,
            self.task.completed,
            self.task.total,
            exc_info=True,
        )


class AdminDatastoreService:
    """Facade exposing Chroma and cache administrative operations."""

    def __init__(self, chroma: ChromaPort | None) -> None:
        if chroma is None:
            raise RuntimeError("Chroma port is not configured.")
        self._chroma = chroma

    # Chroma collection management -------------------------------------------------

    def add_document(self, document: Document) -> Mapping[str, Any]:
        """Upsert a document into the requested collection."""
        logger.info(
            "add_document payload received: %s-%s for collection %s.",
            document.document_id,
            document.name,
            document.collection,
        )
        collection = self._ensure_collection(document.collection)
        collection.upsert(
            ids=[str(document.document_id)],
            documents=[document.text],
            metadatas=[document.metadata],
        )
        return {
            "status": "ok",
            "document_id": document.document_id,
            "doc_name": document.name,
        }

    def _ensure_collection(self, name: str):
        collection = self._chroma.get_collection(name)
        if collection is not None:
            return collection
        self._chroma.create_collection(name)
        collection = self._chroma.get_collection(name)
        if collection is None:
            raise CollectionNotFoundError(f"Collection '{name}' not found after creation")
        return collection

    def create_collection(self, payload: CollectionCreate) -> Mapping[str, Any]:
        """Create a new collection, raising on conflicts or invalid names."""
        self._chroma.create_collection(payload.name)
        return {
            "status": "created",
            "name": payload.name.strip(),
        }

    def delete_collection(self, name: str) -> None:
        """Delete a collection by name."""
        self._chroma.delete_collection(name)

    def list_collections(self) -> list[str]:
        """Return all collection names."""
        return self._chroma.list_collections()

    def count_documents(self, name: str) -> Mapping[str, Any]:
        """Return the number of documents stored in the collection."""
        count = self._chroma.count_documents(name)
        return {"collection": name.strip(), "count": count}

    def list_document_ids(self, name: str) -> Mapping[str, Any]:
        """Return all document ids for the specified collection."""
        ids = self._chroma.list_document_ids(name)
        return {"collection": name.strip(), "count": len(ids), "ids": ids}

    def delete_document(self, name: str, document_id: str) -> None:
        """Delete a specific document from a collection."""
        self._chroma.delete_document(name, document_id)

    def get_document_text(self, name: str, document_id: str) -> Mapping[str, Any]:
        """Return text and metadata for a specific document."""
        text, metadata = self._chroma.get_document_text_and_metadata(name, document_id)
        return {
            "collection": name.strip(),
            "document_id": str(document_id),
            "text": text,
            "metadata": dict(metadata),
        }

    # Merge orchestration ----------------------------------------------------------

    def validate_merge_collections(self, source: str, dest: str) -> None:
        """Ensure both collections exist before initiating a merge."""
        self._chroma.get_collections_pair(source, dest)

    async def run_merge_dry_run(self, dest: str, request: MergeRequest) -> MergeTaskStatus:
        """Execute a dry-run merge in a background thread and return the summary."""
        status = MergeTaskStatus(
            task_id=str(uuid.uuid4()), source=request.source, dest=dest, status="pending"
        )
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            _merge_executor, MergeTaskRunner(self._chroma, status, request).execute
        )
        return status

    async def start_merge(self, dest: str, request: MergeRequest) -> MergeTaskStatus:
        """Start a merge task and return the initial status."""
        task_id = str(uuid.uuid4())
        status = MergeTaskStatus(
            task_id=task_id,
            source=request.source,
            dest=dest,
            status="pending",
        )
        _merge_tasks[task_id] = status
        cancel_event = threading.Event()
        _merge_task_cancel_flags[task_id] = cancel_event

        async def runner() -> None:
            try:
                async with merge_global_semaphore:
                    async with merge_collection_locks[dest]:
                        if cancel_event.is_set():
                            status.status = "cancelled"
                            status.finished_at = time.time()
                            _update_eta_metrics(status)
                            return
                        status.status = "running"
                        loop = asyncio.get_event_loop()
                        await loop.run_in_executor(
                            _merge_executor, MergeTaskRunner(self._chroma, status, request).execute
                        )
            finally:
                _merge_task_cancel_flags.pop(task_id, None)

        asyncio.create_task(runner())
        return status

    def get_merge_status(self, task_id: str) -> MergeTaskStatus:
        """Return the status for a merge task or raise if missing."""
        task = _merge_tasks.get(task_id)
        if task is None:
            raise KeyError(task_id)
        return task

    def cancel_merge(self, task_id: str) -> MergeTaskStatus:
        """Cancel an inflight merge task."""
        task = _merge_tasks.get(task_id)
        if task is None:
            raise KeyError(task_id)
        if task.status not in {"pending", "running"}:
            raise RuntimeError(f"cannot cancel task in status {task.status}")
        cancel_flag = _merge_task_cancel_flags.get(task_id)
        if cancel_flag:
            cancel_flag.set()
        return task

    # Cache administration ---------------------------------------------------------

    def clear_all_caches(self, older_than_days: float | None) -> Mapping[str, Any]:
        """Clear caches across all namespaces, optionally pruning by age."""
        if older_than_days is not None:
            if older_than_days <= 0:
                raise ValueError("older_than_days must be greater than 0")
            cutoff = time.time() - older_than_days * 86400
            removed = cache_manager.prune_all(cutoff)
            return {
                "status": "pruned",
                "cutoff_epoch": cutoff,
                "removed": removed,
            }
        cache_manager.clear_all()
        return {"status": "cleared"}

    def clear_cache(self, name: str, older_than_days: float | None) -> Mapping[str, Any]:
        """Clear or prune a specific cache namespace."""
        if older_than_days is not None:
            if older_than_days <= 0:
                raise ValueError("older_than_days must be greater than 0")
            cutoff = time.time() - older_than_days * 86400
            try:
                removed = cache_manager.prune_cache(name, cutoff)
            except KeyError as exc:
                raise KeyError(f"Cache '{name}' not found") from exc
            return {
                "status": "pruned",
                "cache": name,
                "cutoff_epoch": cutoff,
                "removed": removed,
            }
        try:
            cache_manager.clear_cache(name)
        except KeyError as exc:
            raise KeyError(f"Cache '{name}' not found") from exc
        return {"status": "cleared", "cache": name}

    def cache_stats(self) -> Mapping[str, Any]:
        """Return cache statistics."""
        return cache_manager.stats()

    def inspect_cache(self, name: str, sample_limit: int) -> MutableMapping[str, Any]:
        """Return detailed cache stats for a specific namespace."""
        if sample_limit <= 0:
            raise ValueError("sample_limit must be greater than 0")
        cache = self._get_cache(name)
        details = cache.detailed_stats(sample_limit=sample_limit)
        details["sample_limit"] = sample_limit
        return details

    def _get_cache(self, name: str):
        try:
            return cache_manager.cache(name)
        except KeyError as exc:
            raise KeyError(f"Cache '{name}' not found") from exc


update_eta_metrics = _update_eta_metrics


__all__ = [
    "AdminDatastoreService",
    "CollectionCreate",
    "Document",
    "MergeRequest",
    "MergeTaskStatus",
    "update_eta_metrics",
]
