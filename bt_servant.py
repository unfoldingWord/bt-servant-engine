"""FastAPI entrypoint for webhooks and admin endpoints.

Includes Meta webhook processing and ChromaDB admin utilities.
"""
# pylint: disable=line-too-long,too-many-lines

import asyncio
import concurrent.futures
import json
import time
import hmac
import hashlib
from typing import Optional, Annotated, Any, DefaultDict, cast
from contextlib import asynccontextmanager
from collections import defaultdict
import os
import uuid
from datetime import datetime, timezone
import httpx
from fastapi import FastAPI, Request, Response, status, HTTPException, Header, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from brain import create_brain
from logger import get_logger
from config import config
from utils.perf import (
    set_current_trace,
    time_block,
    record_external_span,
    log_final_report,
)
from db import (
    get_user_chat_history,
    update_user_chat_history,
    get_user_response_language,
    get_or_create_chroma_collection,
    create_chroma_collection,
    delete_chroma_collection,
    delete_document,
    list_chroma_collections,
    count_documents_in_collection,
    get_document_text_and_metadata,
    list_document_ids_in_collection,
    iter_collection_batches,
    get_chroma_collections_pair,
    max_numeric_id_in_collection,
    CollectionExistsError,
    CollectionNotFoundError,
    DocumentNotFoundError,
)
from messaging import (
    send_text_message,
    send_voice_message,
    transcribe_voice_message,
    send_typing_indicator_message,
)
from user_message import UserMessage

@asynccontextmanager
async def _lifespan(_: FastAPI):
    """Initialize shared resources at startup and clean up on shutdown."""
    logger.info("Initializing bt servant engine...")
    logger.info("Loading brain...")
    global brain  # pylint: disable=global-statement
    brain = create_brain()
    # Bump the default thread pool size
    loop = asyncio.get_event_loop()
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=64)
    loop.set_default_executor(executor)
    logger.info("brain loaded.")
    try:
        yield
    finally:
        executor.shutdown(wait=False)


app = FastAPI(lifespan=_lifespan)

brain: Any | None = None

logger = get_logger(__name__)
user_locks: DefaultDict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
merge_global_semaphore = asyncio.Semaphore(1)
merge_collection_locks: DefaultDict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

# Dedicated single-worker executor for merge operations to avoid competing
# with other background work configured on the default executor.
_merge_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

_merge_tasks: dict[str, Any] = {}
_merge_task_cancel_flags: dict[str, asyncio.Event] = {}


# Conservative cap for total input tokens per embeddings request.
# OpenAI currently enforces ~300k tokens per request. Keep a safety margin.
_EMBEDDING_MAX_TOKENS_PER_REQUEST = int(os.environ.get("OPENAI_EMBED_MAX_TOKENS_PER_REQUEST", "290000"))


def _estimate_tokens(text: str | None) -> int:
    """Rough token estimate from text length.

    Heuristic: ~4 chars per token. Always at least 1 to avoid zeros.
    """
    if not text:
        return 1
    # Use max(1, n//4) to be conservative without floating math.
    return max(1, len(text) // 4)


def _yield_token_limited_slices(
    ids: list[str],
    docs: list[str] | None,
    metas: list[dict[str, Any]] | None,
    *,
    max_tokens: int,
) -> list[tuple[list[str], list[str], list[dict[str, Any]] | None]]:
    """Split aligned (ids, docs, metas) into sub-batches under a token budget.

    Returns a list of (ids, docs, metas) tuples ready for add(...).
    Only used when re-embedding (i.e., embeddings are not supplied).
    """
    if not ids:
        return []
    if docs is None:
        # Without documents we cannot estimate tokens; fall back to a single batch.
        return [(ids, [], metas)]

    out: list[tuple[list[str], list[str], list[dict[str, Any]] | None]] = []
    cur_ids: list[str] = []
    cur_docs: list[str] = []
    cur_metas: list[dict[str, Any]] | None = [] if metas is not None else None
    budget = 0

    for i, did in enumerate(ids):
        d = docs[i]
        t = _estimate_tokens(d)
        if cur_ids and budget + t > max_tokens:
            out.append((cur_ids, cur_docs, cur_metas))
            cur_ids = []
            cur_docs = []
            cur_metas = [] if metas is not None else None
            budget = 0
        cur_ids.append(did)
        cur_docs.append(d)
        if metas is not None:
            cur_metas.append(metas[i])  # type: ignore[union-attr]
        budget += t

    if cur_ids:
        out.append((cur_ids, cur_docs, cur_metas))
    return out


def init() -> None:
    """Deprecated startup hook retained for test compatibility.

    Tests may monkeypatch `bt_servant.init` to a no-op. App startup now uses
    FastAPI lifespan instead of on_event.
    """
    return None


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


async def _validate_token(
    *,
    expected: Optional[str],
    authorization: Annotated[Optional[str], Header(alias="Authorization")] = None,
    x_admin_token: Annotated[Optional[str], Header(alias="X-Admin-Token")] = None,
) -> None:
    """Shared helper to validate bearer/X-Admin-Token headers."""
    if not expected:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Token not configured",
                            headers={"WWW-Authenticate": "Bearer"})

    provided = None
    if authorization and authorization.lower().startswith("bearer "):
        provided = authorization.split(" ", 1)[1].strip()
    elif x_admin_token:
        provided = x_admin_token.strip()

    if not provided:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Missing credentials",
                            headers={"WWW-Authenticate": "Bearer"})
    if not hmac.compare_digest(provided, expected):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Invalid credentials",
                            headers={"WWW-Authenticate": "Bearer"})


async def require_admin_token(
    authorization: Annotated[Optional[str], Header(alias="Authorization")] = None,
    x_admin_token: Annotated[Optional[str], Header(alias="X-Admin-Token")] = None,
):
    """Simple admin token guard for non-webhook endpoints."""
    if not config.ENABLE_ADMIN_AUTH:
        return
    await _validate_token(
        expected=config.ADMIN_API_TOKEN,
        authorization=authorization,
        x_admin_token=x_admin_token,
    )


async def require_healthcheck_token(
    authorization: Annotated[Optional[str], Header(alias="Authorization")] = None,
    x_admin_token: Annotated[Optional[str], Header(alias="X-Admin-Token")] = None,
):
    """Token guard specifically for the health check endpoint."""
    if not config.ENABLE_ADMIN_AUTH:
        return
    await _validate_token(
        expected=config.HEALTHCHECK_API_TOKEN,
        authorization=authorization,
        x_admin_token=x_admin_token,
    )


@app.get("/")
def read_root():
    """Health/info endpoint with a short usage message."""
    return {"message": "Welcome to the API. Refer to /docs for available endpoints."}


@app.get("/alive")
async def alive_check(_: None = Depends(require_healthcheck_token)) -> JSONResponse:
    """Authenticated health check endpoint for infrastructure probes."""
    return JSONResponse({"status": "ok", "message": "BT Servant is alive and healthy."})


@app.post("/chroma/add-document")
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
@app.post("/add-document")
async def add_document_alias(document: Document, _: None = Depends(require_admin_token)):
    """Back-compatibility alias for `/chroma/add-document`."""
    return await add_document(document)


@app.post("/chroma/collections")
async def create_collection_endpoint(payload: CollectionCreate, _: None = Depends(require_admin_token)):
    """Create a Chroma collection by name.

    Returns 201 on creation, 409 if it already exists, 400 on invalid name.
    """
    logger.info("create collection request received: %s", payload.model_dump())
    try:
        create_chroma_collection(payload.name)
    except ValueError as ve:
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content={"error": str(ve)})
    except CollectionExistsError:
        return JSONResponse(status_code=status.HTTP_409_CONFLICT, content={"error": "Collection already exists"})
    return JSONResponse(
        status_code=status.HTTP_201_CREATED,
        content={
            "status": "created",
            "name": payload.name.strip(),
        },
    )


@app.delete("/chroma/collections/{name}")
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
        return JSONResponse(status_code=status.HTTP_404_NOT_FOUND, content={"error": "Collection not found"})
    return Response(status_code=status.HTTP_204_NO_CONTENT)

@app.get("/chroma/collections")
async def list_collections_endpoint(_: None = Depends(require_admin_token)):
    """List all Chroma collection names."""
    names = list_chroma_collections()
    return JSONResponse(status_code=status.HTTP_200_OK, content={"collections": names})


@app.get("/chroma/collections/{name}/count")
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
        return JSONResponse(status_code=status.HTTP_404_NOT_FOUND, content={"error": "Collection not found"})
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "collection": name.strip(),
            "count": count,
        },
    )


@app.get("/chroma/collection/{name}/ids")
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
        return JSONResponse(status_code=status.HTTP_404_NOT_FOUND, content={"error": "Collection not found"})
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "collection": name.strip(),
            "count": len(ids),
            "ids": ids,
        },
    )


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _apply_metadata_tags(  # pylint: disable=too-many-arguments
    metadatas: list[dict[str, Any]] | None,
    *,
    enabled: bool,
    tag_key: str,
    source: str,
    task_id: str,
    tag_timestamp: bool,
    source_ids: list[str] | None = None,
    source_id_key: str = "_source_id",
) -> list[dict[str, Any]] | None:
    if not enabled:
        return metadatas
    if metadatas is None:
        return None
    stamped: list[dict[str, Any]] = []
    ts = _now_iso() if tag_timestamp else None
    for idx, md in enumerate(metadatas):
        m = dict(md) if md is not None else {}
        m[tag_key] = source
        m["_merge_task_id"] = task_id
        if ts is not None:
            m["_merged_at"] = ts
        if source_ids is not None and 0 <= idx < len(source_ids):
            # Stamp original source doc id under configurable key
            m[source_id_key] = source_ids[idx]
        stamped.append(m)
    return stamped


def _compute_duplicate_preview(
    source_col: Any,
    dest_col: Any,
    *,
    limit: int,
    batch_size: int,
) -> tuple[bool, list[str]]:
    """Return (duplicates_found, preview_ids_up_to_limit)."""
    preview: list[str] = []
    # Build a set of existing dest ids for fast lookup
    dest_ids: set[str] = set()
    for batch in iter_collection_batches(dest_col, batch_size=10000, include_embeddings=False):
        for _id in batch.get("ids") or []:
            dest_ids.add(str(_id))
    if not dest_ids:
        return False, []
    # Scan source until we collect up to limit duplicates
    for batch in iter_collection_batches(source_col, batch_size=batch_size, include_embeddings=False):
        for _id in batch.get("ids") or []:
            sid = str(_id)
            if sid in dest_ids:
                preview.append(sid)
                if len(preview) >= limit:
                    return True, preview
    return (len(preview) > 0), preview


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


def _merge_worker(  # pylint: disable=too-many-arguments,too-many-locals,too-many-branches,too-many-statements
    task: MergeTaskStatus,
    req: MergeRequest,
) -> None:
    """Synchronous merge worker executed in dedicated thread."""
    try:
        logger.info(
            "merge start: task_id=%s source=%s dest=%s mode=%s create_new_id=%s on_duplicate=%s use_source_embeddings=%s batch_size=%d dry_run=%s",
            task.task_id,
            req.source,
            task.dest,
            req.mode,
            req.create_new_id,
            req.on_duplicate,
            req.use_source_embeddings,
            req.batch_size,
            req.dry_run,
        )
        task.started_at = time.time()
        source_col, dest_col = get_chroma_collections_pair(req.source, task.dest)
        if req.source.strip() == task.dest.strip():
            raise ValueError("source and dest must be different collections")
        # Preflight counts
        try:
            source_count = source_col.count()
        except TypeError:
            # Some clients require no args
            source_count = source_col.count
        try:
            dest_col.count()
        except TypeError:
            _ = dest_col.count
        task.total = int(source_count)

        # Dry-run: compute duplicates preview and/or next id
        if req.dry_run:
            if not req.create_new_id and req.on_duplicate == "fail":
                dup_found, preview = _compute_duplicate_preview(
                    source_col, dest_col, limit=req.duplicates_preview_limit, batch_size=req.batch_size
                )
                task.duplicates_found = dup_found
                task.duplicate_preview = preview
            if req.create_new_id:
                start_id = max_numeric_id_in_collection(task.dest) + 1
                task.next_id_start = start_id
            task.status = "completed"
            task.finished_at = time.time()
            return

        # If on_duplicate=fail and not creating new ids, preflight for any duplicate
        if not req.create_new_id and req.on_duplicate == "fail":
            dup_found, _ = _compute_duplicate_preview(
                source_col, dest_col, limit=1, batch_size=req.batch_size
            )
            if dup_found:
                raise ValueError("Duplicate ids detected; aborting due to on_duplicate=fail")

        # Prepare ID allocator when create_new_id
        next_id = None
        if req.create_new_id:
            next_id = max_numeric_id_in_collection(task.dest) + 1

        # Begin batch processing
        for batch in iter_collection_batches(
            source_col,
            batch_size=req.batch_size,
            include_embeddings=req.use_source_embeddings,
        ):
            # Cancellation check
            cancel_evt = _merge_task_cancel_flags.get(task.task_id)
            if cancel_evt and cancel_evt.is_set():
                task.status = "cancelled"
                task.finished_at = time.time()
                logger.info(
                    "merge cancelled: task_id=%s source=%s dest=%s completed=%d total=%d",
                    task.task_id,
                    req.source,
                    task.dest,
                    task.completed,
                    task.total,
                )
                return

            src_ids = [str(i) for i in (batch.get("ids") or [])]
            documents = cast(list[str] | None, batch.get("documents"))
            metadatas = cast(list[dict[str, Any]] | None, batch.get("metadatas"))
            embeddings = cast(list[list[float]] | None, batch.get("embeddings")) if req.use_source_embeddings else None

            # Determine destination ids
            if req.create_new_id:
                assert next_id is not None
                dest_ids = [str(i) for i in range(next_id, next_id + len(src_ids))]
                next_id += len(src_ids)
            else:
                dest_ids = src_ids

            # On duplicates handling when not create_new_id
            to_add_indexes = list(range(len(dest_ids)))
            if not req.create_new_id and req.on_duplicate in ("skip", "overwrite"):
                # Check which dest ids already exist
                existing = dest_col.get(ids=dest_ids)
                existing_ids = set(str(i) for i in (existing.get("ids") or []))
                if req.on_duplicate == "skip" and existing_ids:
                    to_add_indexes = [i for i, did in enumerate(dest_ids) if did not in existing_ids]
                    task.skipped += len(existing_ids)
                elif req.on_duplicate == "overwrite" and existing_ids:
                    # delete existing before add
                    dest_col.delete(ids=list(existing_ids))
                    task.overwritten += len(existing_ids)

            if not to_add_indexes:
                continue

            add_ids = [dest_ids[i] for i in to_add_indexes]
            add_docs = [documents[i] for i in to_add_indexes] if documents else None
            add_metas = [metadatas[i] for i in to_add_indexes] if metadatas else None
            # Build aligned list of source ids for items we will add
            add_src_ids = [src_ids[i] for i in to_add_indexes]
            add_metas = _apply_metadata_tags(
                add_metas,
                enabled=req.tag_metadata,
                tag_key=req.tag_metadata_key,
                source=req.source,
                task_id=task.task_id,
                tag_timestamp=req.tag_metadata_timestamp,
                source_ids=add_src_ids if req.tag_source_id else None,
                source_id_key=req.tag_source_id_key,
            )
            add_embs = [embeddings[i] for i in to_add_indexes] if embeddings else None

            # Perform add
            if add_embs is not None:
                # Using source embeddings: one add call is sufficient.
                dest_col.add(ids=add_ids, documents=add_docs, metadatas=add_metas, embeddings=add_embs)
                task.completed += len(add_ids)
                _update_eta_metrics(task)
            else:
                # Re-embedding: split into token-limited sub-batches to respect
                # provider max tokens per request.
                sub_batches = _yield_token_limited_slices(
                    add_ids,
                    add_docs if add_docs is not None else [],
                    add_metas,
                    max_tokens=_EMBEDDING_MAX_TOKENS_PER_REQUEST,
                )
                # If heuristic returned empty (shouldn't), fallback to single call
                if not sub_batches:
                    sub_batches = [(add_ids, add_docs or [], add_metas)]
                if len(sub_batches) > 1:
                    logger.info(
                        "split add into %d sub-batches (max_tokens=%d)",
                        len(sub_batches),
                        _EMBEDDING_MAX_TOKENS_PER_REQUEST,
                    )
                for sb_ids, sb_docs, sb_metas in sub_batches:
                    dest_col.add(ids=sb_ids, documents=sb_docs, metadatas=sb_metas)
                    task.completed += len(sb_ids)
                    _update_eta_metrics(task)
            # Log each successful insertion mapping from source -> destination id
            try:
                for i in to_add_indexes:
                    src_id = src_ids[i]
                    dst_id = dest_ids[i]
                    logger.info(
                        "doc_id %s from %s inserted into %s with id %s",
                        src_id,
                        req.source,
                        task.dest,
                        dst_id,
                    )
            except Exception:  # pylint: disable=broad-except
                # Avoid impacting merge on logging issues
                pass


            # Move semantics: delete source docs after successful add
            if req.mode == "move":
                source_col.delete(ids=src_ids)
                task.deleted_from_source += len(src_ids)

            # Optional throttle between batches
            if req.sleep_between_batches_ms > 0:
                # Check for cancellation before and after sleeping
                cancel_evt = _merge_task_cancel_flags.get(task.task_id)
                if cancel_evt and cancel_evt.is_set():
                    task.status = "cancelled"
                    task.finished_at = time.time()
                    _update_eta_metrics(task)
                    logger.info(
                        "merge cancelled: task_id=%s source=%s dest=%s completed=%d total=%d",
                        task.task_id,
                        req.source,
                        task.dest,
                        task.completed,
                        task.total,
                    )
                    return
                time.sleep(max(0.0, req.sleep_between_batches_ms / 1000.0))
                cancel_evt = _merge_task_cancel_flags.get(task.task_id)
                if cancel_evt and cancel_evt.is_set():
                    task.status = "cancelled"
                    task.finished_at = time.time()
                    _update_eta_metrics(task)
                    logger.info(
                        "merge cancelled: task_id=%s source=%s dest=%s completed=%d total=%d",
                        task.task_id,
                        req.source,
                        task.dest,
                        task.completed,
                        task.total,
                    )
                    return

        task.status = "completed"
        task.finished_at = time.time()
        _update_eta_metrics(task)
        logger.info(
            "merge completed: task_id=%s source=%s dest=%s completed=%d skipped=%d overwritten=%d deleted_from_source=%d total=%d",
            task.task_id,
            req.source,
            task.dest,
            task.completed,
            task.skipped,
            task.overwritten,
            task.deleted_from_source,
            task.total,
        )
    except Exception as exc:  # pylint: disable=broad-except
        task.status = "failed"
        task.error = str(exc)
        task.finished_at = time.time()
        _update_eta_metrics(task)
        logger.error(
            "merge failed: task_id=%s source=%s dest=%s error=%s status=%s completed=%d total=%d",
            task.task_id,
            req.source,
            task.dest,
            task.error,
            task.status,
            task.completed,
            task.total,
            exc_info=True,
        )


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
    _merge_task_cancel_flags[task_id] = asyncio.Event()

    async def runner() -> None:
        async with merge_global_semaphore:
            async with merge_collection_locks[dest]:
                status_obj.status = "running"
                # Run in dedicated single-worker pool
                await asyncio.get_event_loop().run_in_executor(_merge_executor, _merge_worker, status_obj, req)

    asyncio.create_task(runner())
    return status_obj


@app.post("/chroma/collections/{dest}/merge")
async def merge_collections_endpoint(  # pylint: disable=too-many-return-statements
    dest: str, payload: MergeRequest, _: None = Depends(require_admin_token)
):
    """Merge one collection into another with background processing and dry-run support."""
    # Quick validation of mode/on_duplicate
    if payload.mode not in ("copy", "move"):
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content={"error": "mode must be 'copy' or 'move'"})
    if payload.on_duplicate not in ("fail", "skip", "overwrite"):
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content={"error": "on_duplicate must be one of 'fail','skip','overwrite'"})
    if payload.source.strip() == dest.strip():
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content={"error": "source and dest must differ"})

    # For dry-run, perform preflight synchronously in thread and return summary
    if payload.dry_run:
        # Reuse the worker but with a temporary task to compute summary only
        temp = MergeTaskStatus(task_id=str(uuid.uuid4()), source=payload.source, dest=dest, status="pending")
        try:
            await asyncio.get_event_loop().run_in_executor(_merge_executor, _merge_worker, temp, payload)
        except CollectionNotFoundError as ce:
            return JSONResponse(status_code=status.HTTP_404_NOT_FOUND, content={"error": str(ce)})
        except ValueError as ve:
            return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content={"error": str(ve)})
        return JSONResponse(status_code=status.HTTP_200_OK, content=temp.model_dump())

    # Non-dry-run: start background task
    try:
        # Validate collections exist up front
        get_chroma_collections_pair(payload.source, dest)
    except CollectionNotFoundError as ce:
        return JSONResponse(status_code=status.HTTP_404_NOT_FOUND, content={"error": str(ce)})
    except ValueError as ve:
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content={"error": str(ve)})

    status_obj = await _start_merge_task(dest, payload)
    return JSONResponse(status_code=status.HTTP_202_ACCEPTED, content=status_obj.model_dump())


@app.get("/chroma/merge-tasks/{task_id}")
async def get_merge_task_status(task_id: str, _: None = Depends(require_admin_token)):
    """Return the status of a merge task by id."""
    status_obj = _merge_tasks.get(task_id)
    if not status_obj:
        return JSONResponse(status_code=status.HTTP_404_NOT_FOUND, content={"error": "task not found"})
    return JSONResponse(status_code=status.HTTP_200_OK, content=status_obj.model_dump())


@app.delete("/chroma/merge-tasks/{task_id}")
async def cancel_merge_task(task_id: str, _: None = Depends(require_admin_token)):
    """Request cancellation of a running/pending merge task."""
    status_obj = _merge_tasks.get(task_id)
    if not status_obj:
        return JSONResponse(status_code=status.HTTP_404_NOT_FOUND, content={"error": "task not found"})
    if status_obj.status not in ("pending", "running"):
        return JSONResponse(status_code=status.HTTP_409_CONFLICT, content={"error": f"cannot cancel task in status {status_obj.status}"})
    cancel_evt = _merge_task_cancel_flags.get(task_id)
    if cancel_evt:
        cancel_evt.set()
    return JSONResponse(status_code=status.HTTP_202_ACCEPTED, content={"status": "cancelling"})


@app.delete("/chroma/collections/{name}/documents/{document_id}")
async def delete_document_endpoint(name: str, document_id: str, _: None = Depends(require_admin_token)):
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
        return JSONResponse(status_code=status.HTTP_404_NOT_FOUND, content={"error": "Collection not found"})
    except DocumentNotFoundError:
        return JSONResponse(status_code=status.HTTP_404_NOT_FOUND, content={"error": "Document not found"})
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@app.get("/chroma/collections/{name}/documents/{document_id}")
async def get_document_text_endpoint(name: str, document_id: str, _: None = Depends(require_admin_token)):
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
        return JSONResponse(status_code=status.HTTP_404_NOT_FOUND, content={"error": "Collection not found"})
    except DocumentNotFoundError:
        return JSONResponse(status_code=status.HTTP_404_NOT_FOUND, content={"error": "Document not found"})
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
@app.api_route("/chroma", methods=[
    "GET", "POST", "DELETE", "PUT", "PATCH", "OPTIONS", "HEAD"
])
async def chroma_root(_: None = Depends(require_admin_token)):
    """Catch-all /chroma root to enforce admin auth and 404 unknown paths."""
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not Found")


@app.api_route("/chroma/{_path:path}", methods=[
    "GET", "POST", "DELETE", "PUT", "PATCH", "OPTIONS", "HEAD"
])
async def chroma_catch_all(_path: str, _: None = Depends(require_admin_token)):
    """Catch-all for any /chroma subpath to enforce admin auth and 404."""
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not Found")


@app.get("/meta-whatsapp")
async def verify_webhook(request: Request):
    """Meta webhook verification endpoint following the standard handshake."""
    params = dict(request.query_params)
    mode = params.get("hub.mode")
    token = params.get("hub.verify_token")
    challenge = params.get("hub.challenge")

    if mode == "subscribe" and token == config.META_VERIFY_TOKEN:
        logger.info("webhook verified successfully with Meta.")
        return Response(content=challenge, media_type="text/plain", status_code=200)
    logger.warning("webhook verification failed.")
    return Response(status_code=403)


@app.post("/meta-whatsapp")
async def handle_meta_webhook(  # pylint: disable=too-many-nested-blocks,too-many-locals,too-many-branches
        request: Request,
        x_hub_signature_256: Annotated[Optional[str], Header(alias="X-Hub-Signature-256")] = None,
        x_hub_signature: Annotated[Optional[str], Header(alias="X-Hub-Signature")] = None,
        user_agent: Annotated[Optional[str], Header(alias="User-Agent")] = None
):
    """Process Meta webhook events: validate signature/UA and dispatch to brain."""
    try:

        body = await request.body()
        # measure signature verification time and attach it to each message trace below
        _sig_t0 = time.time()
        if not verify_facebook_signature(config.META_APP_SECRET, body, x_hub_signature_256, x_hub_signature):
            raise HTTPException(status_code=401, detail="Invalid signature")
        _sig_t1 = time.time()

        if not user_agent or user_agent.strip() != config.FACEBOOK_USER_AGENT:
            logger.error(
                'received invalid user agent: %s. expected: %s',
                user_agent,
                config.FACEBOOK_USER_AGENT,
            )
            raise HTTPException(status_code=401, detail="Invalid User Agent")

        payload = await request.json()
        for entry in payload.get("entry", []):
            for change in entry.get("changes", []):
                value = change.get("value", {})
                messages = value.get("messages", [])
                for message_data in messages:
                    try:
                        user_message = UserMessage.from_data(message_data)
                        # Correlate timing to the specific WhatsApp message id
                        set_current_trace(user_message.message_id)
                        # Attribute earlier signature verification time to this trace
                        record_external_span(
                            name="bt_servant:verify_facebook_signature",
                            start=_sig_t0,
                            end=_sig_t1,
                            trace_id=user_message.message_id,
                        )
                        logger.info("%s message from %s with id %s and timestamp %s received.",
                                    user_message.message_type, user_message.user_id, user_message.message_id,
                                    user_message.timestamp)
                        if not user_message.is_supported_type():
                            logger.warning("unsupported message type: %s received. Skipping message.",
                                           user_message.message_type)
                            continue
                        if user_message.too_old():
                            logger.warning("message %d sec old. dropping old message.", user_message.age())
                            continue
                        if user_message.is_unauthorized_sender():
                            logger.warning("Unauthorized sender: %s", user_message.user_id)
                            return JSONResponse(status_code=status.HTTP_401_UNAUTHORIZED, content={"error": "Unauthorized sender"})

                        # Attribute total handling time per message, including background task duration.
                        _msg_t0 = time.time()
                        # In OpenAI API test mode, run synchronously to avoid background flakiness
                        if os.environ.get("RUN_OPENAI_API_TESTS", "") == "1":
                            try:
                                await process_message(user_message=user_message)
                            finally:
                                record_external_span(
                                    name="bt_servant:handle_meta_webhook",
                                    start=_msg_t0,
                                    end=time.time(),
                                    trace_id=user_message.message_id,
                                )
                        else:
                            task = asyncio.create_task(process_message(user_message=user_message))
                            # Record span when the background task completes
                            def _on_done(_: asyncio.Task, start: float = _msg_t0, trace_id: str = user_message.message_id) -> None:
                                record_external_span(
                                    name="bt_servant:handle_meta_webhook",
                                    start=start,
                                    end=time.time(),
                                    trace_id=trace_id,
                                )
                            task.add_done_callback(_on_done)
                    except ValueError:
                        logger.error("Error while processing user message...", exc_info=True)
                        continue
        return Response(status_code=200)

    except json.JSONDecodeError:
        logger.error("Invalid JSON received", exc_info=True)
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content={"error": "Invalid JSON"})


async def process_message(user_message: UserMessage):  # pylint: disable=too-many-branches
    """Serialize user processing per user id and send responses back."""
    async with user_locks[user_message.user_id]:
        start_time = time.time()
        # ensure all spans produced in this coroutine are associated to this message
        set_current_trace(user_message.message_id)
        # Lazily initialize brain if lifespan didn't run (e.g., certain test harnesses)
        global brain  # pylint: disable=global-statement
        if brain is None:
            logger.warning("Brain not initialized at message time; initializing lazily.")
            brain = create_brain()

        # Top-level guard: ensure any unexpected errors result in a friendly reply
        try:
            async with time_block("bt_servant:process_message"):
                try:
                    await send_typing_indicator_message(user_message.message_id)
                except httpx.HTTPError as e:
                    logger.warning("Failed to send typing indicator: %s", e)

                if user_message.message_type == "audio":
                    text = await transcribe_voice_message(user_message.media_id)
                else:
                    text = user_message.text

                loop = asyncio.get_event_loop()
                assert brain is not None  # mypy: brain set during startup or lazily above
                result = await loop.run_in_executor(None, brain.invoke, {
                    "user_id": user_message.user_id,
                    "user_query": text,
                    "user_chat_history": get_user_chat_history(user_id=user_message.user_id),
                    "user_response_language": get_user_response_language(user_id=user_message.user_id),
                    # Attach perf trace id for cross-thread node timing
                    "perf_trace_id": user_message.message_id,
                })
                responses = result["translated_responses"]
                full_response_text = "\n\n".join(responses).rstrip()
                send_voice = bool(result.get("send_voice_message")) or user_message.message_type == "audio"
                if send_voice:
                    await send_voice_message(user_id=user_message.user_id, text=full_response_text)
                else:
                    response_count = len(responses)
                    if response_count > 1:
                        responses = [f'({i}/{response_count}) {r}' for i, r in enumerate(responses, start=1)]
                    for response in responses:
                        logger.info("Response from bt_servant: %s", response)
                        try:
                            await send_text_message(user_id=user_message.user_id, text=response)
                            await asyncio.sleep(4)
                        except httpx.HTTPError as send_err:
                            logger.error("Failed to send message to Meta for user %s: %s", user_message.user_id, send_err)

                update_user_chat_history(
                    user_id=user_message.user_id,
                    query=user_message.text,
                    response=full_response_text,
                )
        except Exception:  # pylint: disable=broad-except
            # Catch-all for any failure during processing (e.g., upstream rate-limits, unexpected errors)
            logger.error("Unhandled error during process_message; sending fallback to user.", exc_info=True)
            fallback_msg = (
                "It looks like I'm having trouble processing your message. "
                "Please report this issue to my creators."
            )
            try:
                await send_text_message(user_id=user_message.user_id, text=fallback_msg)
            except httpx.HTTPError as send_err:
                logger.error("Failed to send fallback message to Meta for user %s: %s", user_message.user_id, send_err)
        finally:
            logger.info(
                "Overall process_message processing time: %.2f seconds",
                time.time() - start_time,
            )
            # Emit a structured performance report for this message id
            try:
                log_final_report(logger, trace_id=user_message.message_id, user_id=user_message.user_id)
            except Exception:  # pylint: disable=broad-except  # guard logging path
                logger.warning("Failed to emit performance report for message_id=%s", user_message.message_id, exc_info=True)


def verify_facebook_signature(app_secret: str, payload: bytes,
                              sig256: str | None, sig1: str | None) -> bool:
    """
    app_secret: your Meta app secret (string)
    payload: raw request body as bytes
    sig256: value of X-Hub-Signature-256 header (e.g., 'sha256=...')
    sig1:   value of X-Hub-Signature header (e.g., 'sha1=...')  # legacy fallback
    """
    # Prefer SHA-256 if provided
    if sig256:
        expected = "sha256=" + hmac.new(app_secret.encode("utf-8"), payload, hashlib.sha256).hexdigest()
        return hmac.compare_digest(expected, sig256.strip())

    # Fallback to SHA-1 if only that header is present
    if sig1:
        expected = "sha1=" + hmac.new(app_secret.encode("utf-8"), payload, hashlib.sha1).hexdigest()
        return hmac.compare_digest(expected, sig1.strip())

    return False
