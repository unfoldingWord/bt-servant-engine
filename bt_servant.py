import asyncio
import requests
import json
from openai import OpenAI
from collections import defaultdict
from fastapi import FastAPI, Request
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from fastapi import HTTPException
from pathlib import Path
from brain import create_brain
from logger import get_logger
from config import config
from pydantic import BaseModel
from db import add_knowledgebase_doc, delete_knowledgebase_doc, update_knowledgebase_doc

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
    payload = await request.json()
    logger.debug("Received Meta webhook payload: %s", json.dumps(payload, indent=2))

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


@app.post("/insert")
async def insert_entry(entry: KnowledgeBaseEntry):
    try:
        doc_info = {
            "question_or_prompt": entry.question_or_prompt,
            "context_for_expected_response": entry.context_for_expected_response
        }
        doc_text = json.dumps(doc_info)
        logger.info('received new knowledge base doc:\n\n%s', doc_text)
        chroma_id = add_knowledgebase_doc(doc_text)
        logger.info("returning chroma_id: %s", chroma_id)
        return {"knowledgebase_id": chroma_id}
    except Exception as e:
        logger.error("Error while attempting to insert knowledgebase item.", exc_info=True)
        raise HTTPException(status_code=500, detail="Insert failed")


@app.put("/update/{doc_id}")
def update_knowledgebase_entry(doc_id: str, updated_data: KnowledgeBaseEntry):
    try:
        doc_info = {
            "question_or_prompt": updated_data.question_or_prompt,
            "context_for_expected_response": updated_data.context_for_expected_response
        }
        doc_text = json.dumps(doc_info)
        logger.info('received knowledge base doc for update:\n\n%s', doc_text)
        update_knowledgebase_doc(doc_id, doc_text)
        return {"status": "updated"}
    except Exception as e:
        logger.error("Error while attempting to update knowledgebase item.", exc_info=True)
        raise HTTPException(status_code=500, detail="Update failed")


@app.delete("/delete/{doc_id}")
def delete_knowledgebase_entry(doc_id: str):
    try:
        delete_knowledgebase_doc(doc_id)
        return {"status": "deleted"}
    except Exception as e:
        logger.error("Error while attempting to delete knowledgebase item.", exc_info=True)
        raise HTTPException(status_code=500, detail="Deletion failed")


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
