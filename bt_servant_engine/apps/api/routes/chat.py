"""Client-agnostic chat API endpoint."""

from __future__ import annotations

import asyncio
import base64
import os
import tempfile
import time
from contextvars import copy_context
from typing import Any, cast

from fastapi import APIRouter, Depends, HTTPException, Request, status
from openai import OpenAI, OpenAIError

from bt_servant_engine.apps.api.dependencies import require_client_api_key
from bt_servant_engine.apps.api.progress_callback import create_webhook_messenger
from bt_servant_engine.core.api_key_models import APIKey
from bt_servant_engine.apps.api.state import get_brain, set_brain
from bt_servant_engine.apps.api.user_locks import get_user_lock
from bt_servant_engine.core.api_models import ChatRequest, ChatResponse
from bt_servant_engine.core.config import config
from bt_servant_engine.core.logging import get_logger
from bt_servant_engine.core.ports import UserStatePort
from bt_servant_engine.services import ServiceContainer
from bt_servant_engine.services.brain_orchestrator import create_brain
from bt_servant_engine.services.intent_queue import has_queued_intents

router = APIRouter(prefix="/api/v1", tags=["chat"])
logger = get_logger(__name__)

# Voice generation prompt (same as messaging adapter)
VOICE_VIBE_PROMPT = """
Personality/Affect: A knowledgeable and trustworthy guide, providing Scripture readings and translation support with calm confidence.

Voice: Clear, steady, and professional, with a warm and approachable quality, at conversational speaking pace.

Tone: Respectful and engaging, encouraging thoughtful reflection and supporting understanding without distraction.

Dialect: Neutral and standard, avoiding slang or overly casual phrasing; suitable for an international audience.

Pronunciation: Careful and precise, ensuring proper enunciation of biblical names and terms, while remaining natural and fluid.

Features: Uses measured pacing, appropriate pauses, and gentle emphasis to highlight key points. Conveys reverence when reading Scripture and clarity when giving practical instructions.
"""


def _get_services(request: Request) -> ServiceContainer:
    """Retrieve the service container from app state."""
    services = getattr(request.app.state, "services", None)
    if not isinstance(services, ServiceContainer):
        raise RuntimeError("Service container is not configured on app.state.")
    return services


def _require_user_state(services: ServiceContainer) -> UserStatePort:
    """Get UserStatePort from services or raise."""
    user_state = services.user_state
    if user_state is None:
        raise RuntimeError("UserStatePort has not been configured.")
    return user_state


async def _transcribe_audio(audio_base64: str, audio_format: str) -> str:
    """Transcribe base64-encoded audio using OpenAI."""
    # Decode base64 to bytes
    try:
        audio_bytes = base64.b64decode(audio_base64)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid base64 audio data: {exc}",
        ) from exc

    # Write to temp file
    suffix = f".{audio_format}" if audio_format else ".ogg"
    temp_path = os.path.join(tempfile.gettempdir(), f"chat_audio_{time.time()}{suffix}")
    try:
        with open(temp_path, "wb") as f:
            f.write(audio_bytes)

        if os.path.getsize(temp_path) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Audio data is empty",
            )

        # Transcribe using OpenAI
        client = OpenAI(api_key=config.OPENAI_API_KEY)
        with open(temp_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="gpt-4o-transcribe", file=audio_file
            )
        logger.info("Transcribed audio: %s", transcript.text)
        return transcript.text
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


async def _generate_tts(text: str) -> str:
    """Generate TTS audio and return as base64-encoded MP3."""
    temp_path = os.path.join(tempfile.gettempdir(), f"tts_output_{time.time()}.mp3")
    try:
        client = OpenAI(api_key=config.OPENAI_API_KEY)
        loop = asyncio.get_running_loop()

        def _create_speech() -> None:
            with client.audio.speech.with_streaming_response.create(
                model="gpt-4o-mini-tts",
                voice="ash",
                input=text,
                instructions=VOICE_VIBE_PROMPT,
            ) as response:
                response.stream_to_file(temp_path)

        await loop.run_in_executor(None, _create_speech)

        # Read and encode as base64
        with open(temp_path, "rb") as f:
            audio_bytes = f.read()
        return base64.b64encode(audio_bytes).decode("utf-8")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def _compute_agentic_strengths(user_id: str, user_state: UserStatePort) -> tuple[str, str | None]:
    """Return effective agentic strength and stored user preference (if any)."""
    user_strength = user_state.get_agentic_strength(user_id=user_id)
    system_strength = str(config.AGENTIC_STRENGTH).lower()
    if system_strength not in {"normal", "low", "very_low"}:
        system_strength = "normal"
    effective = user_strength or system_strength
    return effective, user_strength


def _compute_dev_agentic_mcp(user_id: str, user_state: UserStatePort) -> tuple[bool, bool | None]:
    """Return effective dev MCP flag and stored user preference (if any)."""
    user_pref = user_state.get_dev_agentic_mcp(user_id=user_id)
    effective = user_pref if user_pref is not None else config.BT_DEV_AGENTIC_MCP
    return effective, user_pref


async def _build_brain_payload(
    chat_request: ChatRequest,
    user_query: str,
    user_state: UserStatePort,
) -> dict[str, Any]:
    """Build the payload dict for brain invocation.

    If progress_callback_url is provided in the request, creates a webhook
    messenger and enables progress messaging.
    """
    effective_agentic, user_agentic = _compute_agentic_strengths(chat_request.user_id, user_state)
    effective_mcp, user_mcp = _compute_dev_agentic_mcp(chat_request.user_id, user_state)

    # Configure progress messaging if callback URL is provided
    progress_enabled = False
    progress_messenger = None
    progress_throttle_seconds = chat_request.progress_throttle_seconds

    if chat_request.progress_callback_url:
        progress_messenger = await create_webhook_messenger(
            callback_url=chat_request.progress_callback_url,
            user_id=chat_request.user_id,
            auth_token=config.ADMIN_API_TOKEN or None,
        )
        progress_enabled = True
        logger.info(
            "Progress messaging enabled for user %s -> %s",
            chat_request.user_id[:8] + "...",
            chat_request.progress_callback_url,
        )

    payload: dict[str, Any] = {
        "user_id": chat_request.user_id,
        "user_query": user_query,
        "user_chat_history": user_state.get_chat_history_for_llm(user_id=chat_request.user_id),
        "user_response_language": user_state.get_response_language(user_id=chat_request.user_id),
        "agentic_strength": effective_agentic,
        "dev_agentic_mcp": effective_mcp,
        "perf_trace_id": f"chat-{chat_request.user_id}-{time.time()}",
        "progress_enabled": progress_enabled,
        "progress_messenger": progress_messenger,
        "progress_throttle_seconds": progress_throttle_seconds,
        "last_progress_time": 0,
    }
    if user_agentic is not None:
        payload["user_agentic_strength"] = user_agentic
    if user_mcp is not None:
        payload["user_dev_agentic_mcp"] = user_mcp
    return payload


def _extract_response_language(
    result: dict[str, Any], user_id: str, user_state: UserStatePort
) -> str:
    """Extract the response language from brain result."""
    return (
        result.get("final_response_language")
        or result.get("user_response_language")
        or user_state.get_response_language(user_id=user_id)
        or result.get("query_language")
        or "en"
    )


def _get_or_create_brain() -> Any:
    """Get or create the brain instance."""
    brain = get_brain()
    if brain is None:
        logger.warning("Brain not initialized; initializing lazily.")
        brain = create_brain()
        set_brain(brain)
    return brain


async def _resolve_user_query(chat_request: ChatRequest) -> str:
    """Resolve the user query from the request, transcribing audio if needed."""
    if chat_request.message_type == "audio":
        if not chat_request.audio_base64:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="audio_base64 is required when message_type is 'audio'",
            )
        return await _transcribe_audio(chat_request.audio_base64, chat_request.audio_format)
    return chat_request.message


async def _maybe_generate_voice(
    result: dict[str, Any], chat_request: ChatRequest, full_response_text: str
) -> str | None:
    """Generate TTS audio if needed, returning base64 or None."""
    send_voice = bool(result.get("send_voice_message")) or chat_request.message_type == "audio"
    if not send_voice:
        return None
    voice_payload = result.get("voice_message_text") or full_response_text
    if not voice_payload:
        return None
    try:
        return await _generate_tts(voice_payload)
    except OpenAIError as exc:
        logger.error("TTS generation failed: %s", exc)
        return None


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: Request,
    chat_request: ChatRequest,
    api_key: APIKey = Depends(require_client_api_key),  # noqa: ARG001
) -> ChatResponse:
    """
    Process a chat message and return responses.

    Handles both text and audio input. For audio input, transcribes first.
    If the brain decides to respond with voice, generates TTS audio.

    Messages from the same user are serialized via per-user locking to prevent
    race conditions when users send multiple messages in quick succession.
    """
    # Acquire per-user lock to serialize concurrent requests from the same user
    user_lock = await get_user_lock(chat_request.user_id)
    async with user_lock:
        user_state = _require_user_state(_get_services(request))
        brain = _get_or_create_brain()
        if brain is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Brain initialization failed",
            )

        user_query = await _resolve_user_query(chat_request)
        if not user_query.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Message is empty",
            )

        # Build brain payload and invoke
        brain_payload = await _build_brain_payload(chat_request, user_query, user_state)
        ctx = copy_context()

        def _invoke_sync() -> dict[str, Any]:
            return cast(dict[str, Any], ctx.run(brain.invoke, cast(Any, brain_payload)))

        result = await asyncio.get_running_loop().run_in_executor(None, _invoke_sync)

        # Extract responses
        responses = list(result.get("translated_responses", []))
        full_response_text = "\n\n".join(responses).rstrip()

        # Update chat history
        original_message = (
            chat_request.message if chat_request.message_type == "text" else user_query
        )
        user_state.append_chat_history(chat_request.user_id, original_message, full_response_text)

        return ChatResponse(
            responses=responses,
            response_language=_extract_response_language(result, chat_request.user_id, user_state),
            voice_audio_base64=await _maybe_generate_voice(
                result, chat_request, full_response_text
            ),
            intent_processed=str(result.get("triggered_intent", "unknown")),
            has_queued_intents=has_queued_intents(chat_request.user_id),
        )
