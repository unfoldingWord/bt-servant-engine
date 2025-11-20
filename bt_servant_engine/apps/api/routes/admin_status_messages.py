"""Administrative routes for managing status message translations."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Path, Response, status
from pydantic import BaseModel, Field

from bt_servant_engine.apps.api.dependencies import require_admin_token
from bt_servant_engine.services import status_messages

router = APIRouter(prefix="/admin/status-messages")

AuthDependency = Annotated[None, Depends(require_admin_token)]


class StatusMessageCacheResponse(BaseModel):
    """All status message translations, keyed by message id and language."""

    messages: dict[str, dict[str, str]]


class StatusMessageTranslationsResponse(BaseModel):
    """Translations for a single status message key."""

    message_key: str
    translations: dict[str, str]


class StatusMessageUpdate(BaseModel):
    """Payload for creating or updating a translation."""

    text: str = Field(..., min_length=1, description="Localized translation text")


class StatusMessagesByLanguageResponse(BaseModel):
    """All message texts for a single language keyed by message id."""

    language: str
    translations: dict[str, str]


@router.get("", response_model=StatusMessageCacheResponse)
async def list_status_messages(_: AuthDependency) -> StatusMessageCacheResponse:
    """Return the complete cache of status message translations."""
    return StatusMessageCacheResponse(messages=status_messages.list_status_message_cache())


@router.get(
    "/{message_key}",
    response_model=StatusMessageTranslationsResponse,
    responses={404: {"description": "Unknown message key"}},
)
async def get_status_message_translations(
    message_key: Annotated[str, Path(min_length=1)],
    _: AuthDependency,
) -> StatusMessageTranslationsResponse:
    """Return translations for a specific message key."""
    try:
        translations = status_messages.get_status_message_translations(message_key)
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Unknown message key: {message_key}"
        ) from exc
    return StatusMessageTranslationsResponse(message_key=message_key, translations=translations)


@router.put(
    "/{message_key}/{language}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_model=None,
    response_class=Response,
    responses={
        400: {"description": "Invalid input"},
        404: {"description": "Unknown message key"},
    },
)
async def upsert_status_message_translation(
    message_key: Annotated[str, Path(min_length=1)],
    language: Annotated[str, Path(min_length=2, max_length=8)],
    payload: StatusMessageUpdate,
    _: AuthDependency,
) -> None:
    """Create or update a translation for the given message key and language."""
    try:
        status_messages.set_status_message_translation(message_key, language, payload.text)
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Unknown message key: {message_key}"
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.delete(
    "/{message_key}/{language}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_model=None,
    response_class=Response,
    responses={
        400: {"description": "Invalid input"},
        404: {"description": "Unknown message key or translation"},
    },
)
async def delete_status_message_translation(
    message_key: Annotated[str, Path(min_length=1)],
    language: Annotated[str, Path(min_length=2, max_length=8)],
    _: AuthDependency,
) -> None:
    """Delete a translation for the given message key and language (English cannot be deleted)."""
    try:
        status_messages.delete_status_message_translation(message_key, language)
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unknown translation for key={message_key}, language={language}",
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.get(
    "/language/{language}",
    response_model=StatusMessagesByLanguageResponse,
)
async def list_status_messages_by_language(
    language: Annotated[str, Path(min_length=2, max_length=8)],
    _: AuthDependency,
) -> StatusMessagesByLanguageResponse:
    """Return all translations for a given language across all message keys."""
    translations = status_messages.list_status_messages_for_language(language)
    return StatusMessagesByLanguageResponse(language=language, translations=translations)


__all__ = ["router"]
