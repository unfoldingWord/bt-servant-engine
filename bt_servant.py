import asyncio
import concurrent.futures
import json
import time
import hmac
import hashlib
from typing import Optional, Annotated
from openai import OpenAI
from collections import defaultdict
from fastapi import FastAPI, Request, Response, status, HTTPException, Header
from fastapi.responses import JSONResponse
from brain import create_brain
from logger import get_logger
from config import config
from db import get_user_chat_history, update_user_chat_history, get_user_response_language
from messaging import send_text_message, send_voice_message, transcribe_voice_message, send_typing_indicator_message
from user_message import UserMessage

app = FastAPI()

open_ai_client = OpenAI(api_key=config.OPENAI_API_KEY)
brain = None

logger = get_logger(__name__)
user_locks = defaultdict(asyncio.Lock)


@app.on_event("startup")
def init():
    logger.info("Initializing bt servant engine...")
    logger.info("Loading brain...")
    global brain
    brain = create_brain()
    # Bump the default thread pool size
    loop = asyncio.get_event_loop()
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=64)
    loop.set_default_executor(executor)
    logger.info("brain loaded.")


@app.get("/")
def read_root():
    return {"message": "Welcome to the API. Refer to /docs for available endpoints."}


@app.get("/meta-whatsapp")
async def verify_webhook(request: Request):
    params = dict(request.query_params)
    mode = params.get("hub.mode")
    token = params.get("hub.verify_token")
    challenge = params.get("hub.challenge")

    if mode == "subscribe" and token == config.META_VERIFY_TOKEN:
        logger.info("webhook verified successfully with Meta.")
        return Response(content=challenge, media_type="text/plain", status_code=200)
    else:
        logger.warning("webhook verification failed.")
        return Response(status_code=403)


@app.post("/meta-whatsapp")
async def handle_meta_webhook(
        request: Request,
        x_hub_signature_256: Annotated[Optional[str], Header(alias="X-Hub-Signature-256")] = None,
        x_hub_signature: Annotated[Optional[str], Header(alias="X-Hub-Signature")] = None,
        user_agent: Annotated[Optional[str], Header(alias="User-Agent")] = None
):
    try:

        body = await request.body()
        if not verify_facebook_signature(config.META_APP_SECRET, body, x_hub_signature_256, x_hub_signature):
            raise HTTPException(status_code=401, detail="Invalid signature")

        if user_agent.strip() != config.FACEBOOK_USER_AGENT:
            logger.error('received invalid user agent: %s. expected: %s', user_agent, config.FACEBOOK_USER_AGENT)
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
                    except Exception as e:
                        logger.error("Error while processing user message...", exc_info=True)
                        continue
        return Response(status_code=200)

    except json.JSONDecodeError:
        logger.error("Invalid JSON received", exc_info=True)
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content={"error": "Invalid JSON"})
    except Exception as e:
        logger.error("Error handling Meta webhook payload", exc_info=True)
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content={"error": "Internal server error"})


async def process_message(user_message: UserMessage):
    async with user_locks[user_message.user_id]:
        try:
            start_time = time.time()
            try:
                await send_typing_indicator_message(user_message.message_id)
            except Exception as e:
                logger.warning("Failed to send typing indicator: %s", e)

            if user_message.message_type == "audio":
                text = await transcribe_voice_message(user_message.media_id)
            else:
                text = user_message.text

            loop = asyncio.get_event_loop()
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
                        # the sleep below is to prevent the (1/3)(3/3)(2/3) situation
                        # in a prod situation we may want to handle this better - IJL
                        await asyncio.sleep(4)
                    except Exception as send_err:
                        logger.error("Failed to send message to Meta for user %s: %s", user_message.user_id, send_err)

            update_user_chat_history(user_id=user_message.user_id, query=user_message.text, response=full_response_text)
            logger.info("Overall process_message processing time: %.2f seconds", time.time() - start_time)
        except Exception as e:
            logger.error("Error handling Meta webhook payload", exc_info=True)
            # TODO: THIS MESSAGE NEEDS TO BE TRANSLATED AT SOME POINT!!! - IJL
            error_message = ("I'm sorry. I'm having a bad day and I'm having trouble responding. "
                             "Please report this issue to my creators.")
            if user_message.message_type == "audio":
                await send_voice_message(user_id=user_message.user_id, text=error_message)
            else:
                await send_text_message(user_id=user_message.user_id, text=error_message)
            update_user_chat_history(user_id=user_message.user_id, query=user_message.text, response=error_message)


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


