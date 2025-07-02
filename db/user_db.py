from tinydb import TinyDB
from config import config


DATA_DIR = config.DATA_DIR
DATA_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH = DATA_DIR / "db.json"
_db = TinyDB(str(DB_PATH))


def get_user_db() -> TinyDB:
    return _db



