import asyncio
import httpx
import concurrent.futures
import json
import time
from datetime import datetime, timezone
from openai import OpenAI
from collections import defaultdict
from fastapi import FastAPI, Request, Response, status
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pathlib import Path
from brain import create_brain
from logger import get_logger
from config import config
from pydantic import BaseModel
from db import get_user_chat_history, update_user_chat_history, get_user_response_language

app = FastAPI()

open_ai_client = OpenAI()
brain = None

AUDIO_DIR = config.DATA_DIR / "audio"
Path(AUDIO_DIR).mkdir(parents=True, exist_ok=True)

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
        logger.info("Webhook verified successfully with Meta.")
        return Response(content=challenge, media_type="text/plain", status_code=200)
    else:
        logger.warning("Webhook verification failed.")
        return Response(status_code=403)


@app.post("/meta-whatsapp")
async def handle_meta_webhook(request: Request):
    try:
        payload = await request.json()
        logger.info("Received request from meta...")

        for entry in payload.get("entry", []):
            for change in entry.get("changes", []):
                value = change.get("value", {})
                messages = value.get("messages", [])
                if not messages:
                    continue
                for msg in messages:
                    message_id = msg.get("id", "")
                    timestamp = msg.get("timestamp", "")
                    if timestamp:
                        message_time_utc = datetime.fromtimestamp(int(timestamp), tz=timezone.utc)
                        message_time_local = message_time_utc.astimezone()
                        logger.info("Message sent at: %s (UTC)", message_time_utc.isoformat())
                        logger.info("Message sent at: %s (Local)", message_time_local.isoformat())
                        message_age_in_seconds = (datetime.now(timezone.utc) - message_time_utc).total_seconds()
                        logger.info("Message age: %d seconds", message_age_in_seconds)
                        if message_age_in_seconds > config.MESSAGE_AGE_CUTOFF_IN_SECONDS:
                            logger.warning("skipping/dropping old message.")
                            continue

                    user_id = msg.get("from", "")
                    logger.info("message %s from %s with id %s and timestamp %s received.",
                                msg.get("text", ""), user_id, msg.get("id", ""), msg.get("timestamp", ""))

                    if not user_id:
                        logger.warning("Missing 'from' field in message: %s", msg)
                        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content={"error": "Missing 'from' field"})

                    if config.IN_META_SANDBOX_MODE and user_id != config.META_SANDBOX_PHONE_NUMBER:
                        logger.warning("Unauthorized sender in sandbox mode: %s", user_id)
                        return JSONResponse(status_code=status.HTTP_401_UNAUTHORIZED, content={"error": "Unauthorized sender"})

                    text = msg.get("text", {}).get("body", "")
                    logger.info(f"Incoming message from {user_id}: {text}")
                    asyncio.create_task(process_message_and_respond(user_id=user_id, message_id=message_id, query=text))

        return Response(status_code=200)

    except json.JSONDecodeError:
        logger.error("Invalid JSON received", exc_info=True)
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content={"error": "Invalid JSON"})

    except Exception as e:
        logger.error("Error handling Meta webhook payload", exc_info=True)
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content={"error": "Internal server error"})


async def process_message_and_respond(user_id: str, message_id: str, query: str):
    async with user_locks[user_id]:
        start_time = time.time()

        try:
            await send_meta_typing_indicator(message_id)
        except Exception as e:
            logger.warning("Failed to send typing indicator: %s", e)

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, brain.invoke, {
            "user_id": user_id,
            "user_query": query,
            "user_chat_history": get_user_chat_history(user_id=user_id),
            "user_response_language": get_user_response_language(user_id=user_id)
        })
        responses = result["responses"]
        response_count = len(responses)
        if response_count > 1:
            responses = [f'({i}/{response_count}) {r}' for i, r in enumerate(responses, start=1)]
        for response in responses:
            logger.info("Response from bt_servant: %s", response)
            try:
                await send_meta_text_message(user_id=user_id, text=response)
                # the sleep below is to prevent the (1/3)(3/3)(2/3) situation
                # in a prod situation we may want to handle this better - IJL
                await asyncio.sleep(4)
            except Exception as send_err:
                logger.error("Failed to send message to Meta for user %s: %s", user_id, send_err)

        update_user_chat_history(user_id=user_id, query=query, response="\n\n".join(responses).rstrip())
        logger.info("Overall process_message_and_respond processing time: %.2f seconds", time.time() - start_time)


async def send_meta_text_message(user_id: str, text: str):
    url = f"https://graph.facebook.com/v18.0/{config.META_PHONE_NUMBER_ID}/messages"
    headers = {
        "Authorization": f"Bearer {config.META_WHATSAPP_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "messaging_product": "whatsapp",
        "to": user_id,
        "type": "text",
        "text": {"body": text}
    }
    async with httpx.AsyncClient() as client:
        response = await client.post(url, headers=headers, json=payload)
        if response.status_code >= 400:
            logger.error("Failed to send Meta message: %s", response.text)
        else:
            logger.info("Sent Meta message to %s: %s", user_id, text)


async def send_meta_typing_indicator(message_id: str):
    url = f"https://graph.facebook.com/v18.0/{config.META_PHONE_NUMBER_ID}/messages"
    headers = {
        "Authorization": f"Bearer {config.META_WHATSAPP_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "messaging_product": "whatsapp",
        "status": "read",
        "message_id": message_id,
        "typing_indicator": {
            "type": "text"
        }
    }
    async with httpx.AsyncClient() as client:
        response = await client.post(url, headers=headers, json=payload)
        if response.status_code >= 400:
            logger.error("Failed to send typing indicator (via read status): %s", response.text)
        else:
            logger.info("Sent typing indicator via message_id=%s", message_id)


app.mount("/audio", StaticFiles(directory=AUDIO_DIR), name="audio")
