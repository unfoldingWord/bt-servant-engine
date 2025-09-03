"""Messaging utilities for WhatsApp (Meta) and audio transcription/tts."""

import os
import tempfile
import time
import datetime
import asyncio
from typing import Mapping, Sequence, Tuple, Optional, IO

import httpx
from openai import OpenAI

from logger import get_logger
from config import config

logger = get_logger(__name__)
open_ai_client = OpenAI(api_key=config.OPENAI_API_KEY)

# Narrow alias for httpx "files" parameter typing to keep annotations readable.
FileItem = Tuple[
    str,
    Tuple[Optional[str], IO[bytes] | bytes | str]
    | Tuple[Optional[str], IO[bytes], Optional[str]],
]
FilesParam = Sequence[FileItem]


async def send_text_message(user_id: str, text: str) -> None:
    """Send a plain text WhatsApp message to a user."""
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


async def send_voice_message(user_id: str, text: str) -> None:
    """Synthesize `text` to speech and send as a WhatsApp audio message."""
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


async def send_typing_indicator_message(message_id: str) -> None:
    """Send a typing indicator (simulated via read status) for a message id."""
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
    """Download and transcribe a voice message by Meta media id."""
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
                raise RuntimeError("Media retrieval failed")

            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None, _write_audio_to_disk, temp_audio_path, response.content
            )

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
    """Return the direct download URL for a Meta media id."""
    url = f"https://graph.facebook.com/v23.0/{media_id}"
    headers = {
        "Authorization": f"Bearer {config.META_WHATSAPP_TOKEN}"
    }
    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers)
        if response.status_code >= 400:
            logger.error("Failed to get media metadata: %s", response.text)
            raise RuntimeError("Media url retrieval failed")
        url = response.json().get("url", "")
        logger.info('url: %s retrieved from meta media endpoint.', url)
        return url


def _create_voice_message(user_id: str, text: str) -> str:
    """Create a temporary MP3 file from `text` using OpenAI TTS and return its path."""
    start_time = time.time()
    logger.info("Preparing to transform text to audio file...")

    # Create a safe temp file with .mp3 suffix (timezone-aware UTC)
    timestamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%S")
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
    """Upload a voice message file to Meta and return the media id (async wrapper)."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _upload_voice_message_sync, audio_file_path)


def _upload_voice_message_sync(audio_file_path: str) -> str:
    """Blocking upload of a voice message file to Meta; returns the media id."""
    if not os.path.exists(audio_file_path):
        raise FileNotFoundError(f"Audio file not found: {audio_file_path}")
    url = f"https://graph.facebook.com/v23.0/{config.META_PHONE_NUMBER_ID}/media"
    headers = {
        "Authorization": f"Bearer {config.META_WHATSAPP_TOKEN}"
    }
    with open(audio_file_path, "rb") as fh:
        # httpx types accept a sequence of (name, (filename, fileobj, content_type)) tuples
        files: FilesParam = [
            ("file", (os.path.basename(audio_file_path), fh, "audio/mpeg")),
            ("messaging_product", (None, "whatsapp")),
        ]
        response = httpx.post(url, headers=headers, files=files)
    if response.status_code >= 400:
        logger.error("Failed to upload voice message: %s", response.text)
        raise RuntimeError(f"Upload failed: {response.status_code} - {response.text}")

    media = response.json()
    media_id = media.get("id") if isinstance(media, Mapping) else None
    if not media_id:
        raise RuntimeError("Upload succeeded but response contained no media ID.")

    logger.info("Successfully uploaded audio. Media ID: %s", media_id)
    return media_id


def _write_audio_to_disk(temp_audio_path: str, content: bytes) -> None:
    """Write binary audio content to disk at the given path."""
    with open(temp_audio_path, "wb") as f:
        f.write(content)
