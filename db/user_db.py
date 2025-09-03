"""TinyDB user database wiring.

Creates the DB in the configured data directory.
"""
from pathlib import Path

from tinydb import TinyDB

from config import config


# Resolve from settings dynamically to satisfy static analysis.
DATA_DIR = Path(getattr(config, "DATA_DIR", Path("/data")))
DATA_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH = DATA_DIR / "db.json"
_db = TinyDB(str(DB_PATH))


def get_user_db() -> TinyDB:
    """Return the process-wide TinyDB instance for users."""
    return _db
