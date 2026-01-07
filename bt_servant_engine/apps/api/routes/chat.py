"""Client-agnostic chat API endpoint."""

from __future__ import annotations

import asyncio
import base64
import os
import tempfile
import time
from contextvars import copy_context
from typing import Any, cast

from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from openai import OpenAI

from bt_servant_engine.apps.api.state import get_brain, set_brain
from bt_servant_engine.core.api_models import ChatRequest, ChatResponse
from bt_servant_engine.core.config import config
from bt_servant_engine.core.logging import get_logger
from bt_servant_engine.core.ports import UserStatePort
from bt_servant_engine.services import ServiceContainer
from bt_servant_engine.services.brain_orchestrator import create_brain
from bt_servant_engine.services.intent_queue import has_queued_intents

router = APIRouter(prefix="/api/v1", tags=["chat"])
logger = get_logger(__name__)

# Expected number of parts in "Bearer <token>" authorization header
_BEARER_AUTH_PARTS = 2

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


async def _verify_api_key(authorization: str | None = Header(default=None)) -> None:
    """Verify the API key from the Authorization header."""
    if not config.ADMIN_API_TOKEN:
        # If no token is configured, allow all requests (dev mode)
        return

    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header",
        )

    # Expect "Bearer <token>"
    parts = authorization.split()
    if len(parts) != _BEARER_AUTH_PARTS or parts[0].lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Authorization header format. Expected 'Bearer <token>'",
        )

    if parts[1] != config.ADMIN_API_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )


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


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: Request,
    chat_request: ChatRequest,
    _: None = Depends(_verify_api_key),
) -> ChatResponse:
    """
    Process a chat message and return responses.

    Handles both text and audio input. For audio input, transcribes first.
    If the brain decides to respond with voice, generates TTS audio.
    """
    services = _get_services(request)
    user_state = _require_user_state(services)

    # Get or create brain
    brain = get_brain()
    if brain is None:
        logger.warning("Brain not initialized; initializing lazily.")
        brain = create_brain()
        set_brain(brain)
    if brain is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Brain initialization failed",
        )

    # Resolve user query (transcribe if audio)
    if chat_request.message_type == "audio":
        if not chat_request.audio_base64:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="audio_base64 is required when message_type is 'audio'",
            )
        user_query = await _transcribe_audio(chat_request.audio_base64, chat_request.audio_format)
    else:
        user_query = chat_request.message

    if not user_query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Message is empty",
        )

    # Compute agentic settings
    effective_agentic_strength, user_agentic_strength = _compute_agentic_strengths(
        chat_request.user_id, user_state
    )
    effective_dev_agentic_mcp, user_dev_agentic_mcp = _compute_dev_agentic_mcp(
        chat_request.user_id, user_state
    )

    # Build brain payload
    # Note: progress_messenger is None for REST API (no real-time messaging)
    brain_payload: dict[str, Any] = {
        "user_id": chat_request.user_id,
        "user_query": user_query,
        "user_chat_history": user_state.get_chat_history(user_id=chat_request.user_id),
        "user_response_language": user_state.get_response_language(user_id=chat_request.user_id),
        "agentic_strength": effective_agentic_strength,
        "dev_agentic_mcp": effective_dev_agentic_mcp,
        "perf_trace_id": f"chat-{chat_request.user_id}-{time.time()}",
        "progress_enabled": False,  # No progress messages for REST API
        "progress_messenger": None,
        "progress_throttle_seconds": 0,
        "last_progress_time": 0,
    }
    if user_agentic_strength is not None:
        brain_payload["user_agentic_strength"] = user_agentic_strength
    if user_dev_agentic_mcp is not None:
        brain_payload["user_dev_agentic_mcp"] = user_dev_agentic_mcp

    # Invoke brain in executor (it's synchronous)
    loop = asyncio.get_running_loop()
    ctx = copy_context()

    def _invoke_sync() -> dict[str, Any]:
        return cast(dict[str, Any], ctx.run(brain.invoke, cast(Any, brain_payload)))

    result = await loop.run_in_executor(None, _invoke_sync)

    # Extract responses
    responses = list(result.get("translated_responses", []))
    full_response_text = "\n\n".join(responses).rstrip()

    # Determine response language
    response_language = (
        result.get("final_response_language")
        or result.get("user_response_language")
        or user_state.get_response_language(user_id=chat_request.user_id)
        or result.get("query_language")
        or "en"
    )

    # Determine if voice output is needed
    send_voice = bool(result.get("send_voice_message")) or (chat_request.message_type == "audio")
    voice_text = result.get("voice_message_text")
    voice_payload = voice_text or full_response_text

    # Generate TTS if needed
    voice_audio_base64: str | None = None
    if send_voice and voice_payload:
        try:
            voice_audio_base64 = await _generate_tts(voice_payload)
        except Exception as exc:
            logger.error("TTS generation failed: %s", exc)
            # Don't fail the whole request, just skip voice

    # Get intent processed
    intent_processed = str(result.get("triggered_intent", "unknown"))

    # Check for queued intents
    queued = has_queued_intents(chat_request.user_id)

    # Update chat history
    user_state.append_chat_history(
        chat_request.user_id,
        chat_request.message if chat_request.message_type == "text" else user_query,
        full_response_text,
    )

    return ChatResponse(
        responses=responses,
        response_language=response_language,
        voice_audio_base64=voice_audio_base64,
        intent_processed=intent_processed,
        has_queued_intents=queued,
    )
