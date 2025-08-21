"""ChromaDB client and collection helpers.

Provides a persistent client under `DATA_DIR` and utilities to
retrieve or create collections, along with small helper utilities.
"""

from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions
from chromadb.api.models import Collection
from config import config
from logger import get_logger

# Pylint struggles to infer pydantic BaseSettings field types.
# Casting to Path makes the type explicit for static analyzers.
DATA_DIR: Path = Path(str(config.DATA_DIR))
DATA_DIR.mkdir(parents=True, exist_ok=True)

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                model_name="text-embedding-ada-002",
                api_key=config.OPENAI_API_KEY
            )
_aquifer_chroma_db = chromadb.PersistentClient(path=str(DATA_DIR))

logger = get_logger(__name__)


def get_or_create_chroma_collection(name: str) -> Collection:
    """Return an existing Chroma collection or create it if missing."""
    existing_collections = [col.name for col in _aquifer_chroma_db.list_collections()]
    if name in existing_collections:
        return _aquifer_chroma_db.get_collection(name=name, embedding_function=openai_ef)
    logger.info("creating chroma collection: %s", name)
    return _aquifer_chroma_db.create_collection(name=name, embedding_function=openai_ef)


def get_chroma_collection(name: str) -> Collection:
    """Return an existing Chroma collection or raise if missing.

    This does not create collections. Use `get_or_create_chroma_collection`
    at ingestion time, and this getter at query time.
    """
    existing_collections = [col.name for col in _aquifer_chroma_db.list_collections()]
    if name in existing_collections:
        return _aquifer_chroma_db.get_collection(name=name, embedding_function=openai_ef)
    return None


def get_next_doc_id(collection: Collection) -> int:
    """Compute the next integer document id based on existing ids.

    This assumes ids are numeric strings; non-numeric ids are ignored.
    """
    results = collection.get(limit=10000)
    if not results["ids"]:
        return 1
    # Convert string IDs to integers and get the max
    int_ids = [int(doc_id) for doc_id in results["ids"] if doc_id.isdigit()]
    return max(int_ids, default=0) + 1
