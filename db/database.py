from tinydb import TinyDB
from config import Config

DB_DIR = Config.DATA_DIR
DB_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH = DB_DIR / "db.json"
_db = TinyDB(str(DB_PATH))


def get_db() -> TinyDB:
    return _db
