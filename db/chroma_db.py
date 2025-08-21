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


class CollectionExistsError(Exception):
    """Raised when attempting to create a collection that already exists."""


class CollectionNotFoundError(Exception):
    """Raised when attempting to access or delete a missing collection."""


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


def list_chroma_collections() -> list[str]:
    """List names of all existing Chroma collections."""
    return [col.name for col in _aquifer_chroma_db.list_collections()]


def _validate_collection_name(name: str) -> str:
    cleaned = name.strip()
    if not cleaned:
        raise ValueError("Collection name must be non-empty")
    return cleaned


def create_chroma_collection(name: str) -> Collection:
    """Create a new Chroma collection, raising if it already exists."""
    cleaned = _validate_collection_name(name)
    existing = list_chroma_collections()
    if cleaned in existing:
        raise CollectionExistsError(f"Collection '{cleaned}' already exists")
    logger.info("creating chroma collection: %s", cleaned)
    return _aquifer_chroma_db.create_collection(name=cleaned, embedding_function=openai_ef)


def delete_chroma_collection(name: str) -> None:
    """Delete a Chroma collection by name, raising if missing."""
    cleaned = _validate_collection_name(name)
    existing = list_chroma_collections()
    if cleaned not in existing:
        raise CollectionNotFoundError(f"Collection '{cleaned}' not found")
    logger.info("deleting chroma collection: %s", cleaned)
    _aquifer_chroma_db.delete_collection(name=cleaned)
