import asyncio
import time
import requests
from openai import OpenAI
from collections import defaultdict
from twilio.rest import Client as TwilioClient
import strawberry
from strawberry.fastapi import GraphQLRouter
from deepgram import DeepgramClient, PrerecordedOptions
from fastapi import FastAPI, BackgroundTasks, Request, Form
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from twilio.base.exceptions import TwilioRestException
from tinydb import TinyDB, Query as TinyQuery
from pathlib import Path
from threading import Lock
from brain import create_brain
from logger import get_logger
from config import Config
from datetime import datetime

twilio_client = TwilioClient()
app = FastAPI()
open_ai_client = OpenAI()
brain = None

AUDIO_DIR = Config.DATA_DIR / "audio"
Path(AUDIO_DIR).mkdir(parents=True, exist_ok=True)

BASE_DIR = Path(__file__).resolve().parent
DB_DIR = Config.DATA_DIR
DB_PATH = DB_DIR / "chat_history.json"
db = TinyDB(str(DB_PATH))
db_lock = Lock()
CHAT_HISTORY_MAX = 5

logger = get_logger(__name__)

user_locks = defaultdict(asyncio.Lock)


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


@strawberry.type
class Query:
    @strawberry.field
    def query_bt_servant(self, query: str) -> str:
        user_id = '+1231231234'
        logger.info("Received message from %s: %s", user_id, query)
        result = brain.invoke({
            "user_query": query,
            "user_chat_history": get_user_chat_history(user_id=user_id)
        })
        responses = result["responses"]
        response = "\n\n".join(responses).rstrip()
        logger.info("Response from bt_servant: %s", response)
        update_user_chat_history(user_id=user_id, query=query, response=response)
        return response

    @strawberry.field
    def health_check(self) -> str:
        return "aquifer_whatsapp_bot is healthy!"


async def process_message_and_respond(user_id: str, query: str, is_voice_msg_sequence: str = True):
    async with user_locks[user_id]:
        start_time = time.time()
        try:
            result = brain.invoke({
                "user_query": query,
                "user_chat_history": get_user_chat_history(user_id=user_id)
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
            logger.info("Total processing time: %.2f seconds", time.time() - start_time)


def create_voice_file_from_text(user_id: str, text: str):
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


def get_user_chat_history(user_id):
    with db_lock:
        q = TinyQuery()
        result = db.get(q.user_id == user_id)
        return result["history"] if result else []


def update_user_chat_history(user_id, query, response):
    with db_lock:
        q = TinyQuery()
        result = db.get(q.user_id == user_id)
        history = result["history"] if result else []
        history.append({
            "user_message": query,
            "assistant_response": response
        })
        history = history[-CHAT_HISTORY_MAX:]
        db.upsert({"user_id": user_id, "history": history}, q.user_id == user_id)


def transcribe_voice_message(media_url: str) -> str:
    auth = (Config.TWILIO_ACCOUNT_SID, Config.TWILIO_AUTH_TOKEN)

    # Try downloading from Twilio with proper auth
    response = requests.get(media_url, auth=auth)

    if response.status_code != 200:
        logger.error("Failed to fetch media. Status: %s %s\nURL: %s", response.status_code, response.reason, media_url)
        logger.error("Response text: %s", response.text)
        response.raise_for_status()

    # Send raw bytes to Deepgram
    dg = DeepgramClient()
    options = PrerecordedOptions(model="nova-3", smart_format=True)
    result = dg.listen.prerecorded.v("1").transcribe_file({"buffer": response.content}, options)

    return result["results"]["channels"][0]["alternatives"][0]["transcript"]


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
        logger.info("Voice message processing time for user %s: %.2f seconds", user_id, time.time() - start_time)


schema = strawberry.Schema(query=Query)
graphql_app = GraphQLRouter(schema)
app.include_router(graphql_app, prefix="/graphql")
app.mount("/audio", StaticFiles(directory=AUDIO_DIR), name="audio")
