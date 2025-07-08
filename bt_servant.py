import asyncio
import requests
import json
from openai import OpenAI
from collections import defaultdict
from fastapi import FastAPI, Request
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from brain import create_brain
from logger import get_logger
from config import config
from pydantic import BaseModel

app = FastAPI()

open_ai_client = OpenAI()
brain = None

AUDIO_DIR = config.DATA_DIR / "audio"
Path(AUDIO_DIR).mkdir(parents=True, exist_ok=True)

logger = get_logger(__name__)
user_locks = defaultdict(asyncio.Lock)


class KnowledgeBaseEntry(BaseModel):
    question_or_prompt: str
    context_for_expected_response: str


@app.on_event("startup")
def init():
    logger.info("Initializing bt servant engine...")
    logger.info("Loading brain...")
    global brain
    brain = create_brain()
    logger.info("brain loaded.")


@app.get("/")
def read_root():
    return {"message": "Welcome to the API. Refer to /docs for available endpoints."}


@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Incoming {request.method} request to {request.url}")
    response = await call_next(request)
    return response


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
    logger.info("inside handle_meta_webhook...");
    payload = await request.json()
    logger.info("Received Meta webhook payload: %s", json.dumps(payload, indent=2))

    try:
        for entry in payload.get("entry", []):
            for change in entry.get("changes", []):
                value = change.get("value", {})
                messages = value.get("messages", [])
                if not messages:
                    continue

                for msg in messages:
                    user_id = msg["from"]
                    text = msg.get("text", {}).get("body", "")
                    logger.info(f"Incoming message from {user_id}: {text}")

                    # Echo it back
                    send_meta_text_message(to=user_id, text=f"You said: {text}")
    except Exception as e:
        logger.error("Error handling Meta webhook payload", exc_info=True)

    return Response(status_code=200)


def send_meta_text_message(to: str, text: str):
    url = f"https://graph.facebook.com/v18.0/{config.META_PHONE_NUMBER_ID}/messages"
    headers = {
        "Authorization": f"Bearer {config.META_WHATSAPP_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "text",
        "text": {"body": text}
    }

    response = requests.post(url, headers=headers, json=payload)
    logger.info("Sent Meta message to %s: %s", to, text)
    if response.status_code >= 400:
        logger.error("Failed to send Meta message: %s", response.text)


app.mount("/audio", StaticFiles(directory=AUDIO_DIR), name="audio")
