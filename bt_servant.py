import asyncio
import time
import requests
import tempfile
import os
import json
from openai import OpenAI
from collections import defaultdict
from twilio.rest import Client as TwilioClient
import strawberry
from strawberry.fastapi import GraphQLRouter
from fastapi import FastAPI, BackgroundTasks, Request, Form
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from twilio.base.exceptions import TwilioRestException
from pathlib import Path
from brain import create_brain
from logger import get_logger
from config import Config
from datetime import datetime
from db import get_user_chat_history, update_user_chat_history, get_user_response_language
from pydantic import BaseModel
from db import add_knowledgebase_doc

twilio_client = TwilioClient()
app = FastAPI()
open_ai_client = OpenAI()
brain = None

AUDIO_DIR = Config.DATA_DIR / "audio"
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
    return {"message": "Go to /graphql to access the GraphQL API"}


@app.post("/whatsapp")
async def whatsapp_webhook(
    request: Request,
    Body: str = Form(...),
    From: str = Form(...),
    NumMedia: str = Form("0"),
    MediaContentType0: str = Form(None),
    MediaUrl0: str = Form(None),
    background_tasks: BackgroundTasks = None
):
    user_id = From.replace("whatsapp:", "")

    if NumMedia != "0" and MediaContentType0 and MediaContentType0.startswith("audio"):
        logger.info("Received voice message from %s: %s", user_id, MediaUrl0)
        background_tasks.add_task(handle_voice_message, user_id, MediaUrl0)
    else:
        query = Body
        logger.info("Received text message from %s: %s", user_id, query)
        background_tasks.add_task(process_message_and_respond, user_id, query)

    return Response(status_code=202)


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
        return {"knowledgebase_id": -1}


@strawberry.type
class Query:
    @strawberry.field
    def query_bt_servant(self, query: str) -> str:
        user_id = '+1231231234'
        logger.info("Received message from %s: %s", user_id, query)
        result = brain.invoke({
            "user_id": user_id,
            "user_query": query,
            "user_chat_history": get_user_chat_history(user_id=user_id),
            "user_response_language": get_user_response_language(user_id=user_id)
        })
        responses = result["responses"]
        response = "\n\n".join(responses).rstrip()
        logger.info("Response from bt_servant: %s", response)
        update_user_chat_history(user_id=user_id, query=query, response=response)
        return response

    @strawberry.field
    def health_check(self) -> str:
        return "aquifer_whatsapp_bot is healthy!"


async def process_message_and_respond(user_id: str, query: str, is_voice_msg_sequence: str = False):
    async with user_locks[user_id]:
        start_time = time.time()
        try:
            result = brain.invoke({
                "user_id": user_id,
                "user_query": query,
                "user_chat_history": get_user_chat_history(user_id=user_id),
                "user_response_language": get_user_response_language(user_id=user_id)
            })
            responses = result["responses"]
            if is_voice_msg_sequence:
                full_response_text = "\n\n".join(responses).rstrip()
                send_whatsapp_voice_message(user_id=user_id, text=full_response_text)
            else:
                response_count = len(responses)
                if response_count > 1:
                    responses = [f'({i}/{response_count}) {r}' for i, r in enumerate(responses, start=1)]
                for response in responses:
                    logger.info("Response from bt_servant: %s", response)
                    send_whatsapp_text_message(user_id=user_id, text=response)
                    # the sleep below is to prevent the (1/3)(3/3)(2/3) situation
                    # in a prod situation we would want to handle this better
                    # using the Twilio delivery webhook - IJL
                    await asyncio.sleep(4)
                update_user_chat_history(user_id=user_id, query=query, response="\n\n".join(responses).rstrip())
        except (TwilioRestException, RuntimeError, ValueError) as e:
            logger.error("Error occurred during background message handling", exc_info=True)
        finally:
            logger.info("Overall process_message_and_respond processing time: %.2f seconds", time.time() - start_time)


def create_voice_file_from_text(user_id: str, text: str):
    start_time = time.time()
    logger.info("Preparing to transform text to audio file...")
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    filename = f"response_{user_id}_{timestamp}.mp3"
    speech_file_path = AUDIO_DIR / filename

    with open_ai_client.audio.speech.with_streaming_response.create(
            model="gpt-4o-mini-tts",
            voice="coral",
            input=text,
            instructions="Speak in a cheerful and positive tone."
    ) as response:
        response.stream_to_file(speech_file_path)

    logger.info("Total processing time: %.2f seconds", time.time() - start_time)
    return f"{Config.PUBLIC_BASE_URL}/audio/{filename}"


def send_whatsapp_voice_message(user_id: str, text: str):
    media_url = create_voice_file_from_text(user_id=user_id, text=text)
    sender = f"whatsapp:{Config.TWILIO_PHONE_NUMBER}"
    recipient = f"whatsapp:{user_id}"
    twilio_client.messages.create(
        media_url=[media_url],
        from_=sender,
        to=recipient
    )
    logger.info("Sent voice message to %s: %s", user_id, media_url)


def send_whatsapp_text_message(user_id: str, text: str):
    sender = "whatsapp:" + Config.TWILIO_PHONE_NUMBER
    recipient = "whatsapp:" + user_id
    logger.info('Preparing to send message from %s to %s.', sender, recipient)
    twilio_client.messages.create(
        body=text,
        from_=sender,
        to=recipient
    )


def transcribe_voice_message(media_url: str) -> str:
    start_time = time.time()
    logger.info("Preparing to transcribe audio file to text...")
    auth = (Config.TWILIO_ACCOUNT_SID, Config.TWILIO_AUTH_TOKEN)

    # Try downloading from Twilio with proper auth
    response = requests.get(media_url, auth=auth)
    print(response.headers.get("Content-Type"))

    if response.status_code != 200:
        logger.error("Failed to fetch media. Status: %s %s\nURL: %s", response.status_code, response.reason, media_url)
        logger.error("Response text: %s", response.text)
        response.raise_for_status()

    temp_audio_path = os.path.join(tempfile.gettempdir(), "temp_audio.ogg")
    with open(temp_audio_path, "wb") as f:
        f.write(response.content)

    if os.path.getsize(temp_audio_path) == 0:
        raise ValueError("Downloaded audio file is empty")

    with open(temp_audio_path, "rb") as audio_file:
        transcript = open_ai_client.audio.transcriptions.create(
            model="gpt-4o-transcribe",
            file=audio_file
        )

    duration = time.time() - start_time
    logger.info("Transcription completed in %.2f seconds", duration)
    return transcript.text


async def handle_voice_message(user_id: str, media_url: str):
    start_time = time.time()
    try:
        logger.info("Transcribing voice message for user %s from URL: %s", user_id, media_url)

        # Transcribe directly from Twilio media URL
        transcript = transcribe_voice_message(media_url)
        logger.info("Transcription for %s: %s", user_id, transcript)

        # Send transcription through your brain pipeline
        await process_message_and_respond(user_id, transcript, True)
    except Exception as e:
        logger.error("Error occurred during background message handling", exc_info=True)
    finally:
        logger.info("Total voice message processing time for user %s: %.2f seconds", user_id, time.time() - start_time)


schema = strawberry.Schema(query=Query)
graphql_app = GraphQLRouter(schema)
app.include_router(graphql_app, prefix="/graphql")
app.mount("/audio", StaticFiles(directory=AUDIO_DIR), name="audio")
