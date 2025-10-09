"""Webhook delivery routes (e.g., Meta WhatsApp)."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import os
import time
from collections import defaultdict
from typing import Annotated, Any, DefaultDict, Optional, cast

import httpx
from fastapi import APIRouter, Header, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse

from bt_servant_engine.adapters.messaging import (
    send_text_message,
    send_typing_indicator_message,
    send_voice_message,
    transcribe_voice_message,
)
from bt_servant_engine.apps.api.state import get_brain, set_brain
from bt_servant_engine.core.config import config
from bt_servant_engine.core.logging import get_logger
from bt_servant_engine.services.brain_orchestrator import create_brain
from bt_servant_engine.core.models import UserMessage
from bt_servant_engine.adapters.user_state import (
    get_user_agentic_strength,
    get_user_chat_history,
    get_user_response_language,
    update_user_chat_history,
)
from utils.identifiers import get_log_safe_user_id
from utils.perf import log_final_report, record_external_span, set_current_trace, time_block

router = APIRouter()
logger = get_logger(__name__)

user_locks: DefaultDict[str, asyncio.Lock] = defaultdict(asyncio.Lock)


def _compute_agentic_strengths(user_id: str) -> tuple[str, Optional[str]]:
    """Return effective agentic strength and stored user preference (if any)."""
    user_strength = get_user_agentic_strength(user_id=user_id)
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


@router.post("/meta-whatsapp")
async def handle_meta_webhook(  # pylint: disable=too-many-nested-blocks,too-many-locals,too-many-branches
    request: Request,
    x_hub_signature_256: Annotated[Optional[str], Header(alias="X-Hub-Signature-256")] = None,
    x_hub_signature: Annotated[Optional[str], Header(alias="X-Hub-Signature")] = None,
    user_agent: Annotated[Optional[str], Header(alias="User-Agent")] = None,
):
    """Process Meta webhook events: validate signature/UA and dispatch to brain."""
    try:
        body = await request.body()
        # measure signature verification time and attach it to each message trace below
        _sig_t0 = time.time()
        if not verify_facebook_signature(
            config.META_APP_SECRET, body, x_hub_signature_256, x_hub_signature
        ):
            raise HTTPException(status_code=401, detail="Invalid signature")
        _sig_t1 = time.time()

        if not user_agent or user_agent.strip() != config.FACEBOOK_USER_AGENT:
            logger.error(
                "received invalid user agent: %s. expected: %s",
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
                        log_user_id = get_log_safe_user_id(
                            user_message.user_id, secret=config.LOG_PSEUDONYM_SECRET
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
                            continue
                        if user_message.too_old():
                            logger.warning(
                                "message %d sec old. dropping old message.", user_message.age()
                            )
                            continue
                        if user_message.is_unauthorized_sender():
                            logger.warning("Unauthorized sender: %s", log_user_id)
                            return JSONResponse(
                                status_code=status.HTTP_401_UNAUTHORIZED,
                                content={"error": "Unauthorized sender"},
                            )

                        # Attribute total handling time per message,
                        # including background task duration.
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
                            def _on_done(
                                _: asyncio.Task,
                                start: float = _msg_t0,
                                trace_id: str = user_message.message_id,
                            ) -> None:
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
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST, content={"error": "Invalid JSON"}
        )


async def process_message(user_message: UserMessage):  # pylint: disable=too-many-branches,too-many-locals,too-many-statements
    """Serialize user processing per user id and send responses back."""
    async with user_locks[user_message.user_id]:
        log_user_id = get_log_safe_user_id(user_message.user_id, secret=config.LOG_PSEUDONYM_SECRET)
        start_time = time.time()
        # ensure all spans produced in this coroutine are associated to this message
        set_current_trace(user_message.message_id)
        # Lazily initialize brain if lifespan didn't run (e.g., certain test harnesses)
        brain_instance = get_brain()
        if brain_instance is None:
            logger.warning("Brain not initialized at message time; initializing lazily.")
            brain_instance = create_brain()
            set_brain(brain_instance)

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
                if brain_instance is None:
                    raise RuntimeError("Brain instance should be initialized before invocation")
                effective_agentic_strength, user_agentic_strength = _compute_agentic_strengths(
                    user_message.user_id
                )

                brain_payload: dict[str, Any] = {
                    "user_id": user_message.user_id,
                    "user_query": text,
                    "user_chat_history": get_user_chat_history(user_id=user_message.user_id),
                    "user_response_language": get_user_response_language(
                        user_id=user_message.user_id
                    ),
                    "agentic_strength": effective_agentic_strength,
                    # Attach perf trace id for cross-thread node timing
                    "perf_trace_id": user_message.message_id,
                }
                if user_agentic_strength is not None:
                    brain_payload["user_agentic_strength"] = user_agentic_strength

                result = await loop.run_in_executor(
                    None,
                    lambda: brain_instance.invoke(cast(Any, brain_payload)),
                )
                responses = list(result["translated_responses"])
                full_response_text = "\n\n".join(responses).rstrip()
                send_voice = (
                    bool(result.get("send_voice_message")) or user_message.message_type == "audio"
                )
                voice_text = result.get("voice_message_text")

                if send_voice:
                    voice_payload = voice_text or full_response_text
                    if voice_payload:
                        await send_voice_message(user_id=user_message.user_id, text=voice_payload)

                should_send_text = True
                if send_voice and voice_text is None and user_message.message_type == "audio":
                    # Preserve legacy behavior for audio conversations without
                    # explicit voice payloads.
                    should_send_text = False

                if should_send_text and responses:
                    response_count = len(responses)
                    formatted_responses = list(responses)
                    if response_count > 1:
                        formatted_responses = [
                            f"({i}/{response_count}) {r}"
                            for i, r in enumerate(formatted_responses, start=1)
                        ]
                    for response in formatted_responses:
                        logger.info("Response from bt_servant: %s", response)
                        try:
                            await send_text_message(user_id=user_message.user_id, text=response)
                            await asyncio.sleep(4)
                        except httpx.HTTPError as send_err:
                            logger.error(
                                "Failed to send message to Meta for user %s: %s",
                                log_user_id,
                                send_err,
                            )

                update_user_chat_history(
                    user_id=user_message.user_id,
                    query=user_message.text,
                    response=full_response_text,
                )
        except Exception:  # pylint: disable=broad-except
            # Catch-all for any failure during processing
            # (e.g., upstream rate-limits, unexpected errors).
            logger.error(
                "Unhandled error during process_message; sending fallback to user.",
                exc_info=True,
            )
            fallback_msg = (
                "It looks like I'm having trouble processing your message. "
                "Please report this issue to my creators."
            )
            try:
                await send_text_message(user_id=user_message.user_id, text=fallback_msg)
            except httpx.HTTPError as send_err:
                logger.error(
                    "Failed to send fallback message to Meta for user %s: %s", log_user_id, send_err
                )
        finally:
            logger.info(
                "Overall process_message processing time: %.2f seconds",
                time.time() - start_time,
            )
            # Emit a structured performance report for this message id
            try:
                log_final_report(logger, trace_id=user_message.message_id, user_id=log_user_id)
            except Exception:  # pylint: disable=broad-except  # guard logging path
                logger.warning(
                    "Failed to emit performance report for message_id=%s",
                    user_message.message_id,
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
