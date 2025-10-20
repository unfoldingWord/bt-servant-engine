"""Administrative routes for Chroma datastore and cache management."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Response, status
from fastapi.responses import JSONResponse

from bt_servant_engine.apps.api.dependencies import (
    get_admin_datastore_service,
    require_admin_token,
)
from bt_servant_engine.core.exceptions import (
    CollectionExistsError,
    CollectionNotFoundError,
    DocumentNotFoundError,
)
from bt_servant_engine.services.admin.datastore import (
    AdminDatastoreService,
    CollectionCreate,
    Document,
    MergeRequest,
)

router = APIRouter(prefix="/admin")

MAX_CACHE_SAMPLE_LIMIT = 100

AuthDependency = Annotated[None, Depends(require_admin_token)]
AdminServiceDependency = Annotated[AdminDatastoreService, Depends(get_admin_datastore_service)]


def _http_error(status_code: int, message: str | dict[str, Any]) -> JSONResponse:
    content = message if isinstance(message, dict) else {"error": message}
    return JSONResponse(status_code=status_code, content=content)


@router.post("/chroma/add-document")
async def add_document(
    document: Document,
    _: AuthDependency,
    service: AdminServiceDependency,
) -> JSONResponse:
    """Accept a document payload for ingestion into Chroma."""
    try:
        payload = service.add_document(document)
    except CollectionNotFoundError as exc:
        return _http_error(status.HTTP_404_NOT_FOUND, str(exc))
    return JSONResponse(status_code=status.HTTP_200_OK, content=dict(payload))


@router.post("/chroma/collections")
async def create_collection_endpoint(
    payload: CollectionCreate,
    _: AuthDependency,
    service: AdminServiceDependency,
) -> JSONResponse:
    """Create a Chroma collection by name."""
    try:
        body = service.create_collection(payload)
    except ValueError as exc:
        return _http_error(status.HTTP_400_BAD_REQUEST, str(exc))
    except CollectionExistsError:
        return _http_error(status.HTTP_409_CONFLICT, "Collection already exists")
    return JSONResponse(status_code=status.HTTP_201_CREATED, content=dict(body))


@router.delete("/chroma/collections/{name}")
async def delete_collection_endpoint(
    name: str,
    _: AuthDependency,
    service: AdminServiceDependency,
) -> Response:
    """Delete a Chroma collection by name."""
    try:
        service.delete_collection(name)
    except ValueError as exc:
        return _http_error(status.HTTP_400_BAD_REQUEST, str(exc))
    except CollectionNotFoundError:
        return _http_error(status.HTTP_404_NOT_FOUND, "Collection not found")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/chroma/collections")
async def list_collections_endpoint(
    _: AuthDependency,
    service: AdminServiceDependency,
) -> JSONResponse:
    """List all Chroma collection names."""
    names = service.list_collections()
    return JSONResponse(status_code=status.HTTP_200_OK, content={"collections": names})


@router.get("/chroma/collections/{name}/count")
async def count_documents_endpoint(
    name: str,
    _: AuthDependency,
    service: AdminServiceDependency,
) -> JSONResponse:
    """Return the number of documents inside a collection."""
    try:
        body = service.count_documents(name)
    except ValueError as exc:
        return _http_error(status.HTTP_400_BAD_REQUEST, str(exc))
    except CollectionNotFoundError:
        return _http_error(status.HTTP_404_NOT_FOUND, "Collection not found")
    return JSONResponse(status_code=status.HTTP_200_OK, content=dict(body))


@router.get("/chroma/collection/{name}/ids")
async def list_document_ids_endpoint(
    name: str,
    _: AuthDependency,
    service: AdminServiceDependency,
) -> JSONResponse:
    """Return document identifiers for the specified collection."""
    try:
        body = service.list_document_ids(name)
    except ValueError as exc:
        return _http_error(status.HTTP_400_BAD_REQUEST, str(exc))
    except CollectionNotFoundError:
        return _http_error(status.HTTP_404_NOT_FOUND, "Collection not found")
    return JSONResponse(status_code=status.HTTP_200_OK, content=dict(body))


def _validate_merge_payload(dest: str, payload: MergeRequest) -> JSONResponse | None:
    if payload.mode not in ("copy", "move"):
        return _http_error(status.HTTP_400_BAD_REQUEST, "mode must be 'copy' or 'move'")
    if payload.on_duplicate not in ("fail", "skip", "overwrite"):
        return _http_error(
            status.HTTP_400_BAD_REQUEST,
            "on_duplicate must be one of 'fail','skip','overwrite'",
        )
    if payload.source.strip() == dest.strip():
        return _http_error(status.HTTP_400_BAD_REQUEST, "source and dest must differ")
    return None


@router.post("/chroma/collections/{dest}/merge")
async def merge_collections_endpoint(
    dest: str,
    payload: MergeRequest,
    _: AuthDependency,
    service: AdminServiceDependency,
) -> JSONResponse:
    """Merge one Chroma collection into another."""
    validation_error = _validate_merge_payload(dest, payload)
    if validation_error is not None:
        return validation_error

    if payload.dry_run:
        try:
            status_obj = await service.run_merge_dry_run(dest, payload)
        except (CollectionNotFoundError, ValueError) as exc:
            code = (
                status.HTTP_404_NOT_FOUND
                if isinstance(exc, CollectionNotFoundError)
                else status.HTTP_400_BAD_REQUEST
            )
            return _http_error(code, str(exc))
        return JSONResponse(status_code=status.HTTP_200_OK, content=status_obj.model_dump())

    try:
        service.validate_merge_collections(payload.source, dest)
    except (CollectionNotFoundError, ValueError) as exc:
        code = (
            status.HTTP_404_NOT_FOUND
            if isinstance(exc, CollectionNotFoundError)
            else status.HTTP_400_BAD_REQUEST
        )
        return _http_error(code, str(exc))

    status_obj = await service.start_merge(dest, payload)
    return JSONResponse(status_code=status.HTTP_202_ACCEPTED, content=status_obj.model_dump())


@router.get("/chroma/merge-tasks/{task_id}")
async def get_merge_task_status(
    task_id: str,
    _: AuthDependency,
    service: AdminServiceDependency,
) -> JSONResponse:
    """Return the status of a merge task by id."""
    try:
        task = service.get_merge_status(task_id)
    except KeyError:
        return _http_error(status.HTTP_404_NOT_FOUND, "task not found")
    return JSONResponse(status_code=status.HTTP_200_OK, content=task.model_dump())


@router.delete("/chroma/merge-tasks/{task_id}")
async def cancel_merge_task(
    task_id: str,
    _: AuthDependency,
    service: AdminServiceDependency,
) -> JSONResponse:
    """Attempt to cancel a running merge task."""
    try:
        service.cancel_merge(task_id)
    except KeyError:
        return _http_error(status.HTTP_404_NOT_FOUND, "task not found")
    except RuntimeError as exc:
        return _http_error(status.HTTP_409_CONFLICT, str(exc))
    return JSONResponse(status_code=status.HTTP_202_ACCEPTED, content={"status": "cancelling"})


@router.delete("/chroma/collections/{name}/documents/{document_id}")
async def delete_document_endpoint(
    name: str,
    document_id: str,
    _: AuthDependency,
    service: AdminServiceDependency,
) -> Response:
    """Delete a specific document from a collection."""
    try:
        service.delete_document(name, document_id)
    except ValueError as exc:
        return _http_error(status.HTTP_400_BAD_REQUEST, str(exc))
    except CollectionNotFoundError:
        return _http_error(status.HTTP_404_NOT_FOUND, "Collection not found")
    except DocumentNotFoundError:
        return _http_error(status.HTTP_404_NOT_FOUND, "Document not found")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/chroma/collections/{name}/documents/{document_id}")
async def get_document_text_endpoint(
    name: str,
    document_id: str,
    _: AuthDependency,
    service: AdminServiceDependency,
) -> JSONResponse:
    """Return the text and metadata for a specific document."""
    try:
        body = service.get_document_text(name, document_id)
    except ValueError as exc:
        return _http_error(status.HTTP_400_BAD_REQUEST, str(exc))
    except CollectionNotFoundError:
        return _http_error(status.HTTP_404_NOT_FOUND, "Collection not found")
    except DocumentNotFoundError:
        return _http_error(status.HTTP_404_NOT_FOUND, "Document not found")
    return JSONResponse(status_code=status.HTTP_200_OK, content=dict(body))


@router.api_route("/chroma", methods=["GET", "POST", "DELETE", "PUT", "PATCH", "OPTIONS", "HEAD"])
async def chroma_root(_: AuthDependency) -> None:
    """Catch-all /admin/chroma root to enforce admin auth and 404 unknown paths."""
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not Found")


@router.api_route(
    "/chroma/{_path:path}",
    methods=["GET", "POST", "DELETE", "PUT", "PATCH", "OPTIONS", "HEAD"],
)
async def chroma_catch_all(_path: str, _: AuthDependency) -> None:
    """Catch-all for any /admin/chroma subpath to enforce admin auth and 404."""
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not Found")


@router.post("/cache/clear")
async def clear_all_caches(
    _: AuthDependency,
    service: AdminServiceDependency,
    older_than_days: float | None = None,
) -> JSONResponse:
    """Clear or prune all registered caches."""
    try:
        body = service.clear_all_caches(older_than_days)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": str(exc)},
        ) from exc
    return JSONResponse(status_code=status.HTTP_200_OK, content=dict(body))


@router.post("/cache/{name}/clear")
async def clear_named_cache(
    name: str,
    _: AuthDependency,
    service: AdminServiceDependency,
    older_than_days: float | None = None,
) -> JSONResponse:
    """Clear or prune a specific cache namespace."""
    if older_than_days is not None and older_than_days <= 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "older_than_days must be greater than 0"},
        )
    try:
        body = service.clear_cache(name, older_than_days)
    except KeyError as exc:
        message = exc.args[0] if exc.args else str(exc)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": message},
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": str(exc)},
        ) from exc
    return JSONResponse(status_code=status.HTTP_200_OK, content=dict(body))


@router.get("/cache/stats")
async def get_cache_stats(
    _: AuthDependency,
    service: AdminServiceDependency,
) -> JSONResponse:
    """Return summary stats for all caches."""
    return JSONResponse(status_code=status.HTTP_200_OK, content=service.cache_stats())


@router.get("/cache/{name}")
async def inspect_cache(
    name: str,
    _: AuthDependency,
    service: AdminServiceDependency,
    sample_limit: int = 10,
) -> JSONResponse:
    """Return detailed stats and samples for a specific cache."""
    if sample_limit <= 0 or sample_limit > MAX_CACHE_SAMPLE_LIMIT:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": f"sample_limit must be between 1 and {MAX_CACHE_SAMPLE_LIMIT}"},
        )
    try:
        body = service.inspect_cache(name, sample_limit)
    except KeyError as exc:
        message = exc.args[0] if exc.args else str(exc)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": message},
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": str(exc)},
        ) from exc
    return JSONResponse(status_code=status.HTTP_200_OK, content=dict(body))


__all__ = ["router"]
