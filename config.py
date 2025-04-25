import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


class Config:
    IS_FLY = os.getenv("FLY_IO") == "1"
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    # Optional override via env var, with a default
    DATA_DIR = Path(os.getenv("DATA_DIR") or ("/data" if IS_FLY else Path(__file__).resolve().parent / "data"))

    TWILIO_PHONE_NUMBER: str = os.environ.get("TWILIO_PHONE_NUMBER")
    if not TWILIO_PHONE_NUMBER:
        raise RuntimeError("Missing required environment variable: TWILIO_PHONE_NUMBER")
