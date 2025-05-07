import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


class Config:
    PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL")

    IS_FLY = os.getenv("FLY_IO") == "1"
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")

    # Optional override via env var, with a default
    DATA_DIR = Path(os.getenv("DATA_DIR") or ("/data" if IS_FLY else Path(__file__).resolve().parent / "data"))

    TWILIO_PHONE_NUMBER: str = os.environ.get("TWILIO_PHONE_NUMBER")
    TWILIO_ACCOUNT_SID: str = os.getenv("TWILIO_ACCOUNT_SID")
    TWILIO_AUTH_TOKEN: str = os.getenv("TWILIO_AUTH_TOKEN")
    ''


