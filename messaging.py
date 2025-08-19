import httpx
import os
import tempfile
import time
from datetime import datetime
import asyncio
from logger import get_logger
from config import config
from openai import OpenAI

logger = get_logger(__name__)
open_ai_client = OpenAI(api_key=config.OPENAI_API_KEY)


async def send_text_message(user_id: str, text: str):
    url = f"https://graph.facebook.com/v23.0/{config.META_PHONE_NUMBER_ID}/messages"
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


async def send_voice_message(user_id: str, text: str):
    loop = asyncio.get_running_loop()
    path_to_voice_message = await loop.run_in_executor(None, _create_voice_message, user_id, text)
    try:
        media_id = await _upload_voice_message(path_to_voice_message)
        url = f"https://graph.facebook.com/v23.0/{config.META_PHONE_NUMBER_ID}/messages"
        headers = {
            "Authorization": f"Bearer {config.META_WHATSAPP_TOKEN}",
            "Content-Type": "application/json"
        }
        payload = {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": user_id,
            "type": "audio",
            "audio": {"id": media_id}
        }
        async with httpx.AsyncClient() as client:
            response = await client.post(url, headers=headers, json=payload)
            if response.status_code >= 400:
                logger.error("Failed to send Meta voice message: %s", response.text)
            else:
                logger.info("Sent Meta voice message to %s.", user_id)
    finally:
        if os.path.exists(path_to_voice_message):
            os.remove(path_to_voice_message)


async def send_typing_indicator_message(message_id: str):
    url = f"https://graph.facebook.com/v23.0/{config.META_PHONE_NUMBER_ID}/messages"
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


async def transcribe_voice_message(media_id: str) -> str:
    voice_message_url = await _get_media_message_url(media_id=media_id)
    headers = {
        "Authorization": f"Bearer {config.META_WHATSAPP_TOKEN}"
    }
    temp_audio_path = os.path.join(tempfile.gettempdir(), "temp_audio.ogg")
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(voice_message_url, headers=headers)
            if response.status_code >= 400:
                logger.error("Failed to download voice message audio file: %s", response.text)
                raise Exception("Media retrieval failed")

            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, _write_audio_to_disk, temp_audio_path, response.content)

        if os.path.getsize(temp_audio_path) == 0:
            raise ValueError("Downloaded audio file is empty")

        with open(temp_audio_path, "rb") as audio_file:
            transcript = open_ai_client.audio.transcriptions.create(
                model="gpt-4o-transcribe",
                file=audio_file
            )
        transcribed_text = transcript.text
        logger.info("transcription from openAi: %s", transcribed_text)
        return transcribed_text
    finally:
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)


async def _get_media_message_url(media_id: str) -> str:
    url = f"https://graph.facebook.com/v23.0/{media_id}"
    headers = {
        "Authorization": f"Bearer {config.META_WHATSAPP_TOKEN}"
    }
    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers)
        if response.status_code >= 400:
            logger.error("Failed to get media metadata: %s", response.text)
            raise Exception("Media url retrieval failed")
        url = response.json().get("url", "")
        logger.info('url: %s retrieved from meta media endpoint.', url)
        return url


def _create_voice_message(user_id: str, text: str) -> str:
    start_time = time.time()
    logger.info("Preparing to transform text to audio file...")

    # Create a safe temp file with .mp3 suffix
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    filename = f"response_{user_id}_{timestamp}.mp3"
    temp_audio_path = os.path.join(tempfile.gettempdir(), filename)

    with open_ai_client.audio.speech.with_streaming_response.create(
            model="gpt-4o-mini-tts",
            voice="coral",
            input=text,
            instructions="Speak in a cheerful and positive tone."
    ) as response:
        response.stream_to_file(temp_audio_path)

    logger.info("Voice message created in %.2f seconds", time.time() - start_time)
    return temp_audio_path  # return path to temp file


async def _upload_voice_message(audio_file_path: str) -> str:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _upload_voice_message_sync, audio_file_path)


def _upload_voice_message_sync(audio_file_path: str) -> str:
    if not os.path.exists(audio_file_path):
        raise FileNotFoundError(f"Audio file not found: {audio_file_path}")
    url = f"https://graph.facebook.com/v23.0/{config.META_PHONE_NUMBER_ID}/media"
    headers = {
        "Authorization": f"Bearer {config.META_WHATSAPP_TOKEN}"
    }
    files = {
        "file": (os.path.basename(audio_file_path), open(audio_file_path, "rb"), "audio/mpeg"),
        "messaging_product": (None, "whatsapp"),
    }
    response = httpx.post(url, headers=headers, files=files)
    if response.status_code >= 400:
        logger.error("Failed to upload voice message: %s", response.text)
        raise Exception(f"Upload failed: {response.status_code} - {response.text}")

    media_id = response.json().get("id")
    if not media_id:
        raise Exception("Upload succeeded but response contained no media ID.")

    logger.info("Successfully uploaded audio. Media ID: %s", media_id)
    return media_id


def _write_audio_to_disk(temp_audio_path: str, content: bytes):
    with open(temp_audio_path, "wb") as f:
        f.write(content)
