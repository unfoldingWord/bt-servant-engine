"""Webhook delivery routes (e.g., Meta WhatsApp)."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import os
import time
from collections import defaultdict
from dataclasses import dataclass
from types import TracebackType
from typing import (
    Annotated,
    Any,
    Awaitable,
    Callable,
    DefaultDict,
    Iterable,
    NamedTuple,
    Optional,
    cast,
)

from contextvars import copy_context

import httpx
from fastapi import APIRouter, Header, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse

from bt_servant_engine.apps.api.state import get_brain, set_brain
from bt_servant_engine.core.config import config
from bt_servant_engine.core.language import language_indicator
from bt_servant_engine.core.logging import (
    bind_client_ip,
    bind_correlation_id,
    bind_log_user_id,
    get_correlation_id,
    get_logger,
    reset_client_ip,
    reset_correlation_id,
    reset_log_user_id,
)
from bt_servant_engine.core.ports import MessagingPort, UserStatePort
from bt_servant_engine.services import ServiceContainer
from bt_servant_engine.services import status_messages
from bt_servant_engine.services.brain_orchestrator import create_brain
from bt_servant_engine.core.models import RequestContext, UserMessage
from utils.identifiers import get_log_safe_user_id
from utils.perf import log_final_report, record_external_span, set_current_trace, time_block

router = APIRouter()
logger = get_logger(__name__)

user_locks: DefaultDict[str, asyncio.Lock] = defaultdict(asyncio.Lock)


def _get_services(request: Request) -> ServiceContainer:
    services = getattr(request.app.state, "services", None)
    if not isinstance(services, ServiceContainer):
        raise RuntimeError("Service container is not configured on app.state.")
    return services


def _require_messaging(services: ServiceContainer) -> MessagingPort:
    messaging = services.messaging
    if messaging is None:
        raise RuntimeError("MessagingPort has not been configured.")
    return messaging


def _require_user_state(services: ServiceContainer) -> UserStatePort:
    user_state = services.user_state
    if user_state is None:
        raise RuntimeError("UserStatePort has not been configured.")
    return user_state


def _compute_agentic_strengths(
    user_id: str, user_state: UserStatePort
) -> tuple[str, Optional[str]]:
    """Return effective agentic strength and stored user preference (if any)."""
    user_strength = user_state.get_agentic_strength(user_id=user_id)
    system_strength = str(config.AGENTIC_STRENGTH).lower()
    if system_strength not in {"normal", "low", "very_low"}:
        system_strength = "normal"
    effective = user_strength or system_strength
    return effective, user_strength


@router.get("/meta-whatsapp")
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


class _SignatureTiming(NamedTuple):
    start: float
    end: float


@router.post("/meta-whatsapp")
async def handle_meta_webhook(
    request: Request,
    x_hub_signature_256: Annotated[Optional[str], Header(alias="X-Hub-Signature-256")] = None,
    x_hub_signature: Annotated[Optional[str], Header(alias="X-Hub-Signature")] = None,
    user_agent: Annotated[Optional[str], Header(alias="User-Agent")] = None,
):
    """Process Meta webhook events: validate signature/UA and dispatch to brain."""
    try:
        body = await request.body()
        signature_timing = _verify_request_signature(
            body,
            x_hub_signature_256=x_hub_signature_256,
            x_hub_signature=x_hub_signature,
        )
        _validate_user_agent(user_agent)

        services = _get_services(request)
        payload = await request.json()
        request_context: RequestContext | None = getattr(request.state, "request_context", None)
        response = await _dispatch_meta_payload(
            payload,
            services=services,
            signature_timing=signature_timing,
            request_context=request_context,
        )
        if response is not None:
            return response
        return Response(status_code=200)
    except json.JSONDecodeError:
        logger.error("Invalid JSON received", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"error": "Invalid JSON"},
        )


def _validate_user_agent(user_agent: Optional[str]) -> None:
    if user_agent and user_agent.strip() == config.FACEBOOK_USER_AGENT:
        return
    logger.error(
        "received invalid user agent: %s. expected: %s",
        user_agent,
        config.FACEBOOK_USER_AGENT,
    )
    raise HTTPException(status_code=401, detail="Invalid User Agent")


def _verify_request_signature(
    body: bytes,
    *,
    x_hub_signature_256: Optional[str],
    x_hub_signature: Optional[str],
) -> _SignatureTiming:
    start = time.time()
    if not verify_facebook_signature(
        config.META_APP_SECRET,
        body,
        x_hub_signature_256,
        x_hub_signature,
    ):
        raise HTTPException(status_code=401, detail="Invalid signature")
    return _SignatureTiming(start=start, end=time.time())


async def _dispatch_meta_payload(
    payload: dict[str, Any],
    *,
    services: ServiceContainer,
    signature_timing: _SignatureTiming,
    request_context: RequestContext | None,
) -> Response | None:
    for message_data in _iter_meta_messages(payload):
        response = await _handle_meta_message(
            message_data,
            services=services,
            signature_timing=signature_timing,
            request_context=request_context,
        )
        if response is not None:
            return response
    return None


def _iter_meta_messages(payload: dict[str, Any]) -> Iterable[dict[str, Any]]:
    for entry in payload.get("entry", []):
        for change in entry.get("changes", []):
            value = change.get("value", {})
            yield from value.get("messages", [])


async def _handle_meta_message(
    message_data: dict[str, Any],
    *,
    services: ServiceContainer,
    signature_timing: _SignatureTiming,
    request_context: RequestContext | None,
) -> Response | None:
    try:
        user_message = UserMessage.from_data(message_data)
    except ValueError:
        logger.error("Error while processing user message...", exc_info=True)
        return None

    set_current_trace(user_message.message_id)
    record_external_span(
        name="bt_servant:verify_facebook_signature",
        start=signature_timing.start,
        end=signature_timing.end,
        trace_id=user_message.message_id,
    )
    log_user_id = get_log_safe_user_id(
        user_message.user_id,
        secret=config.LOG_PSEUDONYM_SECRET,
    )
    logger.info(
        "%s message from %s with id %s and timestamp %s received.",
        user_message.message_type,
        log_user_id,
        user_message.message_id,
        user_message.timestamp,
    )
    if not user_message.is_supported_type():
        logger.warning(
            "unsupported message type: %s received. Skipping message.",
            user_message.message_type,
        )
        return None
    if user_message.too_old():
        logger.warning(
            "message %d sec old. dropping old message.",
            user_message.age(),
        )
        return None
    if user_message.is_unauthorized_sender():
        logger.warning("Unauthorized sender: %s", log_user_id)
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={"error": "Unauthorized sender"},
        )

    await _dispatch_user_message(
        user_message=user_message,
        services=services,
        correlation_id=request_context.correlation_id if request_context else get_correlation_id(),
        client_ip=request_context.client_ip if request_context else None,
    )
    return None


async def _dispatch_user_message(
    *,
    user_message: UserMessage,
    services: ServiceContainer,
    correlation_id: str | None,
    client_ip: str | None,
) -> None:
    start_time = time.time()
    if os.environ.get("RUN_OPENAI_API_TESTS", "") == "1":
        try:
            await process_message(
                user_message=user_message,
                services=services,
                correlation_id=correlation_id,
                client_ip=client_ip,
            )
        finally:
            record_external_span(
                name="bt_servant:handle_meta_webhook",
                start=start_time,
                end=time.time(),
                trace_id=user_message.message_id,
            )
        return

    task = asyncio.create_task(
        process_message(
            user_message=user_message,
            services=services,
            correlation_id=correlation_id,
            client_ip=client_ip,
        )
    )

    def _on_done(_: asyncio.Task) -> None:
        record_external_span(
            name="bt_servant:handle_meta_webhook",
            start=start_time,
            end=time.time(),
            trace_id=user_message.message_id,
        )

    task.add_done_callback(_on_done)


@dataclass(slots=True)
class _MessageProcessingContext:  # pylint: disable=too-many-instance-attributes
    user_message: UserMessage
    messaging: MessagingPort
    user_state: UserStatePort
    log_user_id: str
    brain: Any
    start_time: float
    correlation_id: Optional[str]
    client_ip: Optional[str]


class _ProcessingGuard:
    """Async context manager to handle message processing cleanup and fallback."""

    def __init__(self, context: _MessageProcessingContext) -> None:
        self._context = context

    async def __aenter__(self) -> _MessageProcessingContext:
        return self._context

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool:
        try:
            if exc is None:
                return False
            if isinstance(exc, asyncio.CancelledError):
                return False
            logger.error(
                "Unhandled error during process_message; sending fallback to user.",
                exc_info=True,
            )
            await _handle_processing_failure(self._context)
            return True
        finally:
            _finalize_processing(self._context)


async def process_message(
    user_message: UserMessage,
    services: ServiceContainer,
    *,
    correlation_id: str | None,
    client_ip: str | None,
) -> None:
    """Serialize user processing per user id and send responses back."""
    async with user_locks[user_message.user_id]:
        token_correlation = bind_correlation_id(correlation_id)
        token_client_ip = bind_client_ip(client_ip)
        set_current_trace(user_message.message_id)
        context = _create_processing_context(
            user_message,
            services,
            correlation_id=correlation_id,
            client_ip=client_ip,
        )
        token = bind_log_user_id(context.log_user_id)
        try:
            async with _ProcessingGuard(context):
                await _process_with_brain(context)
        finally:
            reset_log_user_id(token)
            reset_client_ip(token_client_ip)
            reset_correlation_id(token_correlation)


def _create_processing_context(
    user_message: UserMessage,
    services: ServiceContainer,
    *,
    correlation_id: str | None,
    client_ip: str | None,
) -> _MessageProcessingContext:
    messaging = _require_messaging(services)
    user_state = _require_user_state(services)
    log_user_id = get_log_safe_user_id(
        user_message.user_id,
        secret=config.LOG_PSEUDONYM_SECRET,
    )
    brain_instance = get_brain()
    if brain_instance is None:
        logger.warning("Brain not initialized at message time; initializing lazily.")
        brain_instance = create_brain()
        set_brain(brain_instance)
    if brain_instance is None:
        raise RuntimeError("Brain instance should be initialized before invocation")
    return _MessageProcessingContext(
        user_message=user_message,
        messaging=messaging,
        user_state=user_state,
        log_user_id=log_user_id,
        brain=brain_instance,
        start_time=time.time(),
        correlation_id=correlation_id,
        client_ip=client_ip,
    )


async def _process_with_brain(context: _MessageProcessingContext) -> None:
    async with time_block("bt_servant:process_message"):
        await _send_typing_indicator(context)
        progress_sender = _build_progress_sender(context)
        user_query = await _resolve_user_text(context, progress_sender)
        result = await _invoke_brain(context, user_query, progress_sender)
        full_response_text = await _deliver_responses(context, result, progress_sender)
        context.user_state.append_chat_history(
            context.user_message.user_id,
            context.user_message.text,
            full_response_text,
        )


async def _send_typing_indicator(context: _MessageProcessingContext) -> None:
    try:
        await context.messaging.send_typing_indicator(context.user_message.message_id)
    except httpx.HTTPError as exc:
        logger.warning("Failed to send typing indicator: %s", exc)


def _build_progress_sender(
    context: _MessageProcessingContext,
) -> Callable[[status_messages.LocalizedProgressMessage], Awaitable[None]]:
    async def _send(message: status_messages.LocalizedProgressMessage) -> None:
        if not config.PROGRESS_MESSAGES_ENABLED:
            return
        try:
            text_msg = message.get("text", "")
            if not text_msg:
                logger.debug("Empty progress message text, skipping send")
                return
            await context.messaging.send_text_message(
                context.user_message.user_id,
                text_msg,
            )
        except httpx.HTTPError:
            logger.warning("Failed to send progress message", exc_info=True)

    return _send


async def _resolve_user_text(
    context: _MessageProcessingContext,
    progress_sender: Callable[[status_messages.LocalizedProgressMessage], Awaitable[None]],
) -> str:
    if context.user_message.message_type != "audio":
        return context.user_message.text
    minimal_state = {
        "user_response_language": context.user_state.get_response_language(
            user_id=context.user_message.user_id
        )
    }
    transcribe_msg = status_messages.get_progress_message(
        status_messages.TRANSCRIBING_VOICE,
        minimal_state,
    )
    await progress_sender(transcribe_msg)
    return await context.messaging.transcribe_voice_message(context.user_message.media_id)


async def _invoke_brain(
    context: _MessageProcessingContext,
    user_query: str,
    progress_sender: Callable[[status_messages.LocalizedProgressMessage], Awaitable[None]],
) -> dict[str, Any]:
    effective_agentic_strength, user_agentic_strength = _compute_agentic_strengths(
        context.user_message.user_id,
        context.user_state,
    )
    brain_payload: dict[str, Any] = {
        "user_id": context.user_message.user_id,
        "user_query": user_query,
        "user_chat_history": context.user_state.get_chat_history(
            user_id=context.user_message.user_id
        ),
        "user_response_language": context.user_state.get_response_language(
            user_id=context.user_message.user_id
        ),
        "agentic_strength": effective_agentic_strength,
        "perf_trace_id": context.user_message.message_id,
        "progress_enabled": config.PROGRESS_MESSAGES_ENABLED,
        "progress_messenger": progress_sender,
        "progress_throttle_seconds": config.PROGRESS_MESSAGE_MIN_INTERVAL,
        "last_progress_time": 0,
    }
    if user_agentic_strength is not None:
        brain_payload["user_agentic_strength"] = user_agentic_strength
    loop = asyncio.get_event_loop()
    ctx = copy_context()

    def _invoke_sync() -> dict[str, Any]:
        return cast(dict[str, Any], ctx.run(context.brain.invoke, cast(Any, brain_payload)))

    return await loop.run_in_executor(None, _invoke_sync)


async def _deliver_responses(
    context: _MessageProcessingContext,
    result: dict[str, Any],
    progress_sender: Callable[[status_messages.LocalizedProgressMessage], Awaitable[None]],
) -> str:
    responses = list(result["translated_responses"])
    full_response_text = "\n\n".join(responses).rstrip()
    send_voice = bool(result.get("send_voice_message")) or (
        context.user_message.message_type == "audio"
    )
    voice_text = result.get("voice_message_text")
    if send_voice:
        if voice_text or full_response_text:
            packaging_msg = status_messages.get_progress_message(
                status_messages.PACKAGING_VOICE_RESPONSE,
                result,
            )
            await progress_sender(packaging_msg)
        voice_payload = voice_text or full_response_text
        if voice_payload:
            await context.messaging.send_voice_message(
                context.user_message.user_id,
                voice_payload,
            )
    should_send_text = True
    if send_voice and voice_text is None and context.user_message.message_type == "audio":
        should_send_text = False
    if should_send_text and responses:
        indicator = _response_language_indicator(result, context)
        for response in _format_indicator_responses(responses, indicator):
            logger.info("Response from bt_servant: %s", response)
            try:
                await context.messaging.send_text_message(
                    context.user_message.user_id,
                    response,
                )
                await asyncio.sleep(4)
            except httpx.HTTPError as send_err:
                logger.error(
                    "Failed to send message to Meta for user %s: %s",
                    context.log_user_id,
                    send_err,
                )
    return full_response_text


def _response_language_indicator(result: dict[str, Any], context: _MessageProcessingContext) -> str:
    language = cast(Optional[str], result.get("final_response_language"))
    if not language:
        language = cast(Optional[str], result.get("user_response_language"))
    if not language:
        language = context.user_state.get_response_language(user_id=context.user_message.user_id)
    if not language:
        language = cast(Optional[str], result.get("query_language"))
    return language_indicator(language)


def _format_indicator_responses(responses: list[str], indicator: str) -> list[str]:
    total = len(responses)
    formatted: list[str] = []
    for idx, response in enumerate(responses, start=1):
        prefix = f"({idx}/{total}) " if total > 1 else ""
        formatted.append(f"{indicator} {prefix}{response}")
    return formatted


async def _handle_processing_failure(context: _MessageProcessingContext) -> None:
    error_state = {
        "user_response_language": context.user_state.get_response_language(
            user_id=context.user_message.user_id
        )
    }
    fallback_msg = status_messages.get_status_message(
        status_messages.PROCESSING_ERROR,
        error_state,
    )
    indicator = language_indicator(error_state["user_response_language"])
    fallback_payload = f"{indicator} {fallback_msg}"
    try:
        await context.messaging.send_text_message(
            context.user_message.user_id,
            fallback_payload,
        )
    except httpx.HTTPError as send_err:
        logger.error(
            "Failed to send fallback message to Meta for user %s: %s",
            context.log_user_id,
            send_err,
        )


def _finalize_processing(context: _MessageProcessingContext) -> None:
    logger.info(
        "Overall process_message processing time: %.2f seconds",
        time.time() - context.start_time,
    )
    try:
        log_final_report(
            logger,
            trace_id=context.user_message.message_id,
            user_id=context.log_user_id,
        )
    except (RuntimeError, ValueError, TypeError, KeyError):
        logger.warning(
            "Failed to emit performance report for message_id=%s",
            context.user_message.message_id,
            exc_info=True,
        )


def verify_facebook_signature(
    app_secret: str, payload: bytes, sig256: str | None, sig1: str | None
) -> bool:
    """
    app_secret: your Meta app secret (string)
    payload: raw request body as bytes
    sig256: value of X-Hub-Signature-256 header (e.g., 'sha256=...')
    sig1:   value of X-Hub-Signature header (e.g., 'sha1=...')  # legacy fallback
    """
    # Prefer SHA-256 if provided
    if sig256:
        expected = (
            "sha256=" + hmac.new(app_secret.encode("utf-8"), payload, hashlib.sha256).hexdigest()
        )
        return hmac.compare_digest(expected, sig256.strip())

    # Fallback to SHA-1 if only that header is present
    if sig1:
        expected = "sha1=" + hmac.new(app_secret.encode("utf-8"), payload, hashlib.sha1).hexdigest()
        return hmac.compare_digest(expected, sig1.strip())

    return False


__all__ = ["router"]
