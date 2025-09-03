"""FastAPI entrypoint for webhooks and admin endpoints.

Includes Meta webhook processing and ChromaDB admin utilities.
"""
# pylint: disable=line-too-long

import asyncio
import concurrent.futures
import json
import time
import hmac
import hashlib
from typing import Optional, Annotated, Any, DefaultDict
from contextlib import asynccontextmanager
from collections import defaultdict
import httpx
from fastapi import FastAPI, Request, Response, status, HTTPException, Header, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from brain import create_brain
from logger import get_logger
from config import config
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
    get_document_text,
    list_document_ids_in_collection,
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


async def require_admin_token(
    authorization: Annotated[Optional[str], Header(alias="Authorization")] = None,
    x_admin_token: Annotated[Optional[str], Header(alias="X-Admin-Token")] = None,
):
    """Simple admin token guard for non-webhook endpoints.

    - If ENABLE_ADMIN_AUTH is False, bypass checks (use only for dev/tests).
    - Accepts either `Authorization: Bearer <token>` or `X-Admin-Token: <token>`.
    - Returns 401 if token missing/invalid or not configured.
    """
    if not config.ENABLE_ADMIN_AUTH:
        return
    expected = config.ADMIN_API_TOKEN
    if not expected:
        # Fail-safe if auth is enabled but no token configured
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Admin token not configured",
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


@app.get("/")
def read_root():
    """Health/info endpoint with a short usage message."""
    return {"message": "Welcome to the API. Refer to /docs for available endpoints."}


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
        text = get_document_text(name, document_id)
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
async def handle_meta_webhook(  # pylint: disable=too-many-nested-blocks
        request: Request,
        x_hub_signature_256: Annotated[Optional[str], Header(alias="X-Hub-Signature-256")] = None,
        x_hub_signature: Annotated[Optional[str], Header(alias="X-Hub-Signature")] = None,
        user_agent: Annotated[Optional[str], Header(alias="User-Agent")] = None
):
    """Process Meta webhook events: validate signature/UA and dispatch to brain."""
    try:

        body = await request.body()
        if not verify_facebook_signature(config.META_APP_SECRET, body, x_hub_signature_256, x_hub_signature):
            raise HTTPException(status_code=401, detail="Invalid signature")

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

                        asyncio.create_task(process_message(user_message=user_message))
                    except ValueError:
                        logger.error("Error while processing user message...", exc_info=True)
                        continue
        return Response(status_code=200)

    except json.JSONDecodeError:
        logger.error("Invalid JSON received", exc_info=True)
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content={"error": "Invalid JSON"})


async def process_message(user_message: UserMessage):
    """Serialize user processing per user id and send responses back."""
    async with user_locks[user_message.user_id]:
        start_time = time.time()
        try:
            await send_typing_indicator_message(user_message.message_id)
        except httpx.HTTPError as e:
            logger.warning("Failed to send typing indicator: %s", e)

        if user_message.message_type == "audio":
            text = await transcribe_voice_message(user_message.media_id)
        else:
            text = user_message.text

        loop = asyncio.get_event_loop()
        assert brain is not None  # mypy: brain set during startup
        result = await loop.run_in_executor(None, brain.invoke, {
            "user_id": user_message.user_id,
            "user_query": text,
            "user_chat_history": get_user_chat_history(user_id=user_message.user_id),
            "user_response_language": get_user_response_language(user_id=user_message.user_id)
        })
        responses = result["translated_responses"]
        full_response_text = "\n\n".join(responses).rstrip()
        if user_message.message_type == "audio":
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
        logger.info(
            "Overall process_message processing time: %.2f seconds",
            time.time() - start_time,
        )


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
