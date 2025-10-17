"""Administrative routes for datastore management."""

from __future__ import annotations

import asyncio
import concurrent.futures
import os
import sys
import threading
import time
import uuid
from collections import defaultdict
from typing import Any, DefaultDict, NoReturn, cast

from fastapi import APIRouter, Depends, HTTPException, Response, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from bt_servant_engine.apps.api.dependencies import require_admin_token
from bt_servant_engine.apps.api.routes import admin_merge_helpers as merge_helpers
from bt_servant_engine.core.exceptions import (
    CollectionExistsError,
    CollectionNotFoundError,
    DocumentNotFoundError,
)
from bt_servant_engine.core.logging import get_logger
from bt_servant_engine.services import runtime
from bt_servant_engine.services.cache_manager import cache_manager

router = APIRouter()
logger = get_logger(__name__)

_EMBEDDING_MAX_TOKENS_PER_REQUEST = int(
    os.environ.get("OPENAI_EMBED_MAX_TOKENS_PER_REQUEST", "290000")
)

MAX_CACHE_SAMPLE_LIMIT = 100

# Re-export merge helpers for compatibility with existing references.
iter_collection_batches = merge_helpers.iter_collection_batches
estimate_tokens = merge_helpers.estimate_tokens
yield_token_limited_slices = merge_helpers.yield_token_limited_slices
now_iso = merge_helpers.now_iso
MetadataTaggingConfig = merge_helpers.MetadataTaggingConfig
apply_metadata_tags = merge_helpers.apply_metadata_tags
compute_duplicate_preview = merge_helpers.compute_duplicate_preview


def _chroma_port():
    """Return the configured Chroma port from the service container."""
    services = runtime.get_services()
    if services.chroma is None:
        raise RuntimeError("Chroma port is not configured.")
    return services.chroma


def list_chroma_collections() -> list[str]:
    """List available Chroma collection names."""
    return _chroma_port().list_collections()


def create_chroma_collection(name: str) -> None:
    """Create a new Chroma collection."""
    _chroma_port().create_collection(name)


def delete_chroma_collection(name: str) -> None:
    """Delete an existing Chroma collection if present."""
    _chroma_port().delete_collection(name)


def delete_document(name: str, document_id: str) -> None:
    """Delete a document from the specified collection."""
    _chroma_port().delete_document(name, document_id)


def count_documents_in_collection(name: str) -> int:
    """Return the number of documents stored in a collection."""
    return _chroma_port().count_documents(name)


def get_document_text_and_metadata(name: str, document_id: str) -> tuple[str, dict[str, Any]]:
    """Fetch text and metadata for a stored document."""
    text, metadata = _chroma_port().get_document_text_and_metadata(name, document_id)
    return text, dict(metadata)


def list_document_ids_in_collection(name: str) -> list[str]:
    """Return document identifiers for the collection."""
    return _chroma_port().list_document_ids(name)


def max_numeric_id_in_collection(name: str) -> int:
    """Return the maximum numeric identifier stored in the collection."""
    chroma = _chroma_port()
    collection = chroma.get_collection(name)
    if collection is None:
        raise CollectionNotFoundError(f"Collection '{name}' not found")
    max_id = 0
    for batch in iter_collection_batches(collection, batch_size=10000):
        ids = batch.get("ids") or []
        for value in ids:
            if isinstance(value, str) and value.isdigit():
                max_id = max(max_id, int(value))
    return max_id


def get_chroma_collections_pair(source: str, dest: str) -> tuple[Any, Any]:
    """Return collection handles for a source/destination pair."""
    return _chroma_port().get_collections_pair(source, dest)


def get_or_create_chroma_collection(name: str) -> Any:
    """Return collection handle, creating the collection if absent."""
    chroma = _chroma_port()
    collection = chroma.get_collection(name)
    if collection is not None:
        return collection
    chroma.create_collection(name)
    collection = chroma.get_collection(name)
    if collection is None:
        raise CollectionNotFoundError(f"Collection '{name}' not found")
    return collection


def _raise_error(status_code: int, message: str) -> NoReturn:
    """Raise an HTTPException with the canonical error payload."""
    raise HTTPException(status_code=status_code, detail={"error": message})


merge_global_semaphore = asyncio.Semaphore(1)
merge_collection_locks: DefaultDict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

# Dedicated single-worker executor for merge operations to avoid competing
# with other background work configured on the default executor.
_merge_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

_merge_tasks: dict[str, "MergeTaskStatus"] = {}
_merge_task_cancel_flags: dict[str, threading.Event] = {}


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


@router.post("/chroma/add-document")
async def add_document(document: Document, _: None = Depends(require_admin_token)):
    """Accepts a document payload for future ingestion into Chroma.

    For now, simply logs the received payload.
    """
    logger.info(
        "add_document payload received: %s-%s for collection %s.",
        document.document_id,
        document.name,
        document.collection,
    )
    # Upsert into ChromaDB
    collection = get_or_create_chroma_collection(document.collection)
    collection.upsert(
        ids=[str(document.document_id)],
        documents=[document.text],
        metadatas=[document.metadata],
    )

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "status": "ok",
            "document_id": document.document_id,
            "doc_name": document.name,
        },
    )


# Back-compatibility alias used by tests and earlier clients
@router.post("/add-document")
async def add_document_alias(document: Document, _: None = Depends(require_admin_token)):
    """Back-compatibility alias for `/chroma/add-document`."""
    return await add_document(document)


@router.post("/chroma/collections")
async def create_collection_endpoint(
    payload: CollectionCreate, _: None = Depends(require_admin_token)
):
    """Create a Chroma collection by name.

    Returns 201 on creation, 409 if it already exists, 400 on invalid name.
    """
    logger.info("create collection request received: %s", payload.model_dump())
    try:
        create_chroma_collection(payload.name)
    except ValueError as ve:
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content={"error": str(ve)})
    except CollectionExistsError:
        return JSONResponse(
            status_code=status.HTTP_409_CONFLICT, content={"error": "Collection already exists"}
        )
    return JSONResponse(
        status_code=status.HTTP_201_CREATED,
        content={
            "status": "created",
            "name": payload.name.strip(),
        },
    )


@router.delete("/chroma/collections/{name}")
async def delete_collection_endpoint(name: str, _: None = Depends(require_admin_token)):
    """Delete a Chroma collection by name.

    Returns 204 on success, 404 if missing, 400 on invalid name.
    """
    logger.info("delete collection request received: %s", name)
    try:
        delete_chroma_collection(name)
    except ValueError as ve:
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content={"error": str(ve)})
    except CollectionNotFoundError:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND, content={"error": "Collection not found"}
        )
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/chroma/collections")
async def list_collections_endpoint(_: None = Depends(require_admin_token)):
    """List all Chroma collection names."""
    names = list_chroma_collections()
    return JSONResponse(status_code=status.HTTP_200_OK, content={"collections": names})


@router.get("/chroma/collections/{name}/count")
async def count_documents_endpoint(name: str, _: None = Depends(require_admin_token)):
    """Return the number of documents in the specified collection.

    Returns 200 with `{ "collection": name, "count": n }` on success,
    404 if the collection does not exist, and 400 on invalid name.
    """
    logger.info("count documents request received: %s", name)
    try:
        count = count_documents_in_collection(name)
    except ValueError as ve:
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content={"error": str(ve)})
    except CollectionNotFoundError:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND, content={"error": "Collection not found"}
        )
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "collection": name.strip(),
            "count": count,
        },
    )


@router.get("/chroma/collection/{name}/ids")
async def list_document_ids_endpoint(name: str, _: None = Depends(require_admin_token)):
    """Return all document ids for the specified collection.

    Response format:
    { "collection": name, "count": N, "ids": ["id1", ...] }
    """
    logger.info("list document ids request received: %s", name)
    try:
        ids = list_document_ids_in_collection(name)
    except ValueError as ve:
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content={"error": str(ve)})
    except CollectionNotFoundError:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND, content={"error": "Collection not found"}
        )
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "collection": name.strip(),
            "count": len(ids),
            "ids": ids,
        },
    )


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


class MergeRunner:
    """Orchestrate a merge operation while keeping helper methods small."""

    def __init__(self, task: MergeTaskStatus, req: MergeRequest) -> None:
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
        self.source_col, self.dest_col = get_chroma_collections_pair(
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
            start_id = max_numeric_id_in_collection(self.task.dest) + 1
            self.task.next_id_start = start_id
        self.task.status = "completed"
        self.task.finished_at = time.time()

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
            self.next_id = max_numeric_id_in_collection(self.task.dest) + 1

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


def _merge_worker(task: MergeTaskStatus, req: MergeRequest) -> None:
    """Synchronous merge worker executed in dedicated thread."""
    MergeRunner(task, req).execute()


# Public aliases for test coverage without accessing private helpers directly.
update_eta_metrics = _update_eta_metrics
merge_worker = _merge_worker


async def _start_merge_task(dest: str, req: MergeRequest) -> MergeTaskStatus:
    """Create and start a merge task; returns the initial status."""
    task_id = str(uuid.uuid4())
    status_obj = MergeTaskStatus(
        task_id=task_id,
        source=req.source,
        dest=dest,
        status="pending",
        total=0,
        completed=0,
        skipped=0,
        overwritten=0,
        deleted_from_source=0,
        duplicates_found=None,
        duplicate_preview=None,
        error=None,
        started_at=None,
        finished_at=None,
    )
    _merge_tasks[task_id] = status_obj
    _merge_task_cancel_flags[task_id] = threading.Event()

    async def runner() -> None:
        async with merge_global_semaphore:
            async with merge_collection_locks[dest]:
                status_obj.status = "running"
                # Run in dedicated single-worker pool
                await asyncio.get_event_loop().run_in_executor(
                    _merge_executor, _merge_worker, status_obj, req
                )

    asyncio.create_task(runner())
    return status_obj


@router.post("/chroma/collections/{dest}/merge")
async def merge_collections_endpoint(
    dest: str, payload: MergeRequest, _: None = Depends(require_admin_token)
):
    """Merge one collection into another with background processing and dry-run support."""
    # Quick validation of mode/on_duplicate
    if payload.mode not in ("copy", "move"):
        _raise_error(status.HTTP_400_BAD_REQUEST, "mode must be 'copy' or 'move'")
    if payload.on_duplicate not in ("fail", "skip", "overwrite"):
        _raise_error(
            status.HTTP_400_BAD_REQUEST,
            "on_duplicate must be one of 'fail','skip','overwrite'",
        )
    if payload.source.strip() == dest.strip():
        _raise_error(status.HTTP_400_BAD_REQUEST, "source and dest must differ")

    # For dry-run, perform preflight synchronously in thread and return summary
    if payload.dry_run:
        # Reuse the worker but with a temporary task to compute summary only
        temp = MergeTaskStatus(
            task_id=str(uuid.uuid4()), source=payload.source, dest=dest, status="pending"
        )
        try:
            await asyncio.get_event_loop().run_in_executor(
                _merge_executor, _merge_worker, temp, payload
            )
        except CollectionNotFoundError as ce:
            _raise_error(status.HTTP_404_NOT_FOUND, str(ce))
        except ValueError as ve:
            _raise_error(status.HTTP_400_BAD_REQUEST, str(ve))
        return JSONResponse(status_code=status.HTTP_200_OK, content=temp.model_dump())

    # Non-dry-run: start background task
    try:
        # Validate collections exist up front
        get_chroma_collections_pair(payload.source, dest)
    except CollectionNotFoundError as ce:
        _raise_error(status.HTTP_404_NOT_FOUND, str(ce))
    except ValueError as ve:
        _raise_error(status.HTTP_400_BAD_REQUEST, str(ve))

    status_obj = await _start_merge_task(dest, payload)
    return JSONResponse(status_code=status.HTTP_202_ACCEPTED, content=status_obj.model_dump())


@router.get("/chroma/merge-tasks/{task_id}")
async def get_merge_task_status(task_id: str, _: None = Depends(require_admin_token)):
    """Return the status of a merge task by id."""
    status_obj = _merge_tasks.get(task_id)
    if not status_obj:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND, content={"error": "task not found"}
        )
    return JSONResponse(status_code=status.HTTP_200_OK, content=status_obj.model_dump())


@router.delete("/chroma/merge-tasks/{task_id}")
async def cancel_merge_task(task_id: str, _: None = Depends(require_admin_token)):
    """Request cancellation of a running/pending merge task."""
    status_obj = _merge_tasks.get(task_id)
    if not status_obj:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND, content={"error": "task not found"}
        )
    if status_obj.status not in ("pending", "running"):
        return JSONResponse(
            status_code=status.HTTP_409_CONFLICT,
            content={"error": f"cannot cancel task in status {status_obj.status}"},
        )
    cancel_evt = _merge_task_cancel_flags.get(task_id)
    if cancel_evt:
        cancel_evt.set()
    return JSONResponse(status_code=status.HTTP_202_ACCEPTED, content={"status": "cancelling"})


@router.delete("/chroma/collections/{name}/documents/{document_id}")
async def delete_document_endpoint(
    name: str, document_id: str, _: None = Depends(require_admin_token)
):
    """Delete a specific document from a collection by id.

    Returns 204 on success, 404 if missing, 400 on invalid inputs.
    """
    logger.info(
        "delete document request received: collection=%s, id=%s",
        name,
        document_id,
    )
    try:
        delete_document(name, document_id)
    except ValueError as ve:
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content={"error": str(ve)})
    except CollectionNotFoundError:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND, content={"error": "Collection not found"}
        )
    except DocumentNotFoundError:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND, content={"error": "Document not found"}
        )
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/chroma/collections/{name}/documents/{document_id}")
async def get_document_text_endpoint(
    name: str, document_id: str, _: None = Depends(require_admin_token)
):
    """Return the text of a specific document from a collection by id.

    Returns 200 with `{"collection": name, "document_id": id, "text": "..."}` on success,
    404 if the collection or document does not exist, and 400 on invalid inputs.
    """
    logger.info(
        "get document text request received: collection=%s, id=%s",
        name,
        document_id,
    )
    try:
        text, metadata = get_document_text_and_metadata(name, document_id)
    except ValueError as ve:
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content={"error": str(ve)})
    except CollectionNotFoundError:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND, content={"error": "Collection not found"}
        )
    except DocumentNotFoundError:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND, content={"error": "Document not found"}
        )
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "collection": name.strip(),
            "document_id": str(document_id),
            "text": text,
            "metadata": metadata,
        },
    )


# Catch-all routes under /chroma to enforce auth on unknown paths
@router.api_route("/chroma", methods=["GET", "POST", "DELETE", "PUT", "PATCH", "OPTIONS", "HEAD"])
async def chroma_root(_: None = Depends(require_admin_token)):
    """Catch-all /chroma root to enforce admin auth and 404 unknown paths."""
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not Found")


@router.api_route(
    "/chroma/{_path:path}", methods=["GET", "POST", "DELETE", "PUT", "PATCH", "OPTIONS", "HEAD"]
)
async def chroma_catch_all(_path: str, _: None = Depends(require_admin_token)):
    """Catch-all for any /chroma subpath to enforce admin auth and 404."""
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not Found")


__all__ = ["router"]


@router.post("/cache/clear")
async def clear_all_caches(
    older_than_days: float | None = None,
    _: None = Depends(require_admin_token),
):
    """Clear all registered caches."""
    if older_than_days is not None:
        if older_than_days <= 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"error": "older_than_days must be greater than 0"},
            )
        cutoff = time.time() - older_than_days * 86400
        logger.info(
            "[cache-admin] pruning all caches older than %.2f days (cutoff=%s)",
            older_than_days,
            cutoff,
        )
        removed = cache_manager.prune_all(cutoff)
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "status": "pruned",
                "cutoff_epoch": cutoff,
                "removed": removed,
            },
        )
    logger.info("[cache-admin] clearing all caches via admin endpoint")
    cache_manager.clear_all()
    return JSONResponse(status_code=status.HTTP_200_OK, content={"status": "cleared"})


@router.post("/cache/{name}/clear")
async def clear_named_cache(
    name: str,
    older_than_days: float | None = None,
    _: None = Depends(require_admin_token),
):
    """Clear a specific cache namespace."""
    if older_than_days is not None:
        if older_than_days <= 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"error": "older_than_days must be greater than 0"},
            )
        cutoff = time.time() - older_than_days * 86400
        try:
            removed = cache_manager.prune_cache(name, cutoff)
        except KeyError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"error": f"Cache '{name}' not found"},
            ) from exc
        logger.info(
            "[cache-admin] pruned cache %s older than %.2f days (cutoff=%s, removed=%d)",
            name,
            older_than_days,
            cutoff,
            removed,
        )
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "status": "pruned",
                "cache": name,
                "cutoff_epoch": cutoff,
                "removed": removed,
            },
        )
    try:
        cache_manager.clear_cache(name)
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail={"error": f"Cache '{name}' not found"}
        ) from exc
    logger.info("[cache-admin] cleared cache %s via admin endpoint", name)
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={"status": "cleared", "cache": name},
    )


@router.get("/cache/stats")
async def get_cache_stats(_: None = Depends(require_admin_token)):
    """Return summary stats for all caches."""
    data = cache_manager.stats()
    return JSONResponse(status_code=status.HTTP_200_OK, content=data)


@router.get("/cache/{name}")
async def inspect_cache(name: str, sample_limit: int = 10, _: None = Depends(require_admin_token)):
    """Return detailed stats and samples for a specific cache."""
    if sample_limit <= 0 or sample_limit > MAX_CACHE_SAMPLE_LIMIT:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": f"sample_limit must be between 1 and {MAX_CACHE_SAMPLE_LIMIT}",
            },
        )
    try:
        cache = cache_manager.cache(name)
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail={"error": f"Cache '{name}' not found"}
        ) from exc
    logger.info("[cache-admin] inspecting cache %s (limit=%d)", name, sample_limit)
    details = cache.detailed_stats(sample_limit=sample_limit)
    details["sample_limit"] = sample_limit
    return JSONResponse(status_code=status.HTTP_200_OK, content=details)
