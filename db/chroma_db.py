"""ChromaDB client and collection helpers.

Provides a persistent client under `DATA_DIR` and utilities to
retrieve or create collections, along with small helper utilities.
"""

from pathlib import Path
from typing import Any, Optional, Iterator, Sequence, Tuple, cast
import chromadb
from chromadb.utils import embedding_functions
from chromadb.config import Settings
from config import config
from logger import get_logger

# Pylint struggles to infer pydantic BaseSettings field types.
# Casting to Path makes the type explicit for static analyzers.
DATA_DIR: Path = Path(str(config.DATA_DIR))
DATA_DIR.mkdir(parents=True, exist_ok=True)

openai_ef: Any = embedding_functions.OpenAIEmbeddingFunction(
                model_name="text-embedding-ada-002",
                api_key=config.OPENAI_API_KEY
            )

settings = Settings(
    chroma_segment_cache_policy="LRU",
    chroma_memory_limit_bytes=1000000000  # ~1GB
)
_aquifer_chroma_db = chromadb.PersistentClient(path=str(DATA_DIR), settings=settings)

logger = get_logger(__name__)


class CollectionExistsError(Exception):
    """Raised when attempting to create a collection that already exists."""


class CollectionNotFoundError(Exception):
    """Raised when attempting to access or delete a missing collection."""


class DocumentNotFoundError(Exception):
    """Raised when attempting to delete a missing document from a collection."""


def get_or_create_chroma_collection(name: str) -> Any:
    """Return an existing Chroma collection or create it if missing."""
    existing_collections = [col.name for col in _aquifer_chroma_db.list_collections()]
    if name in existing_collections:
        return _aquifer_chroma_db.get_collection(name=name, embedding_function=openai_ef)
    logger.info("creating chroma collection: %s", name)
    return _aquifer_chroma_db.create_collection(name=name, embedding_function=openai_ef)


def get_chroma_collection(name: str) -> Optional[Any]:
    """Return an existing Chroma collection or raise if missing.

    This does not create collections. Use `get_or_create_chroma_collection`
    at ingestion time, and this getter at query time.
    """
    existing_collections = [col.name for col in _aquifer_chroma_db.list_collections()]
    if name in existing_collections:
        return _aquifer_chroma_db.get_collection(name=name, embedding_function=openai_ef)
    return None


def get_next_doc_id(collection: Any) -> int:
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


def count_documents_in_collection(name: str) -> int:
    """Return the number of documents in the given collection.

    Raises CollectionNotFoundError if the collection does not exist
    and ValueError if the name is empty/invalid.
    """
    cleaned = _validate_collection_name(name)
    existing = list_chroma_collections()
    if cleaned not in existing:
        raise CollectionNotFoundError(f"Collection '{cleaned}' not found")
    collection = _aquifer_chroma_db.get_collection(name=cleaned, embedding_function=openai_ef)
    return collection.count()


def _validate_collection_name(name: str) -> str:
    cleaned = name.strip()
    if not cleaned:
        raise ValueError("Collection name must be non-empty")
    return cleaned


def create_chroma_collection(name: str) -> Any:
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


def delete_document(collection_name: str, document_id: str) -> None:
    """Delete a document by id from a specific collection.

    Raises CollectionNotFoundError if the collection doesn't exist, and
    DocumentNotFoundError if the id isn't present.
    """
    col_name = _validate_collection_name(collection_name)
    doc_id = str(document_id).strip()
    if not doc_id:
        raise ValueError("Document id must be non-empty")

    existing = list_chroma_collections()
    if col_name not in existing:
        raise CollectionNotFoundError(f"Collection '{col_name}' not found")

    collection = _aquifer_chroma_db.get_collection(name=col_name, embedding_function=openai_ef)
    result = collection.get(ids=[doc_id])
    if not result["ids"]:
        raise DocumentNotFoundError(f"Document '{doc_id}' not found in collection '{col_name}'")
    collection.delete(ids=[doc_id])


def get_document_text(collection_name: str, document_id: str) -> str:
    """Return the text content of a document in a specific collection.

    Raises CollectionNotFoundError if the collection doesn't exist, and
    DocumentNotFoundError if the id isn't present.
    """
    col_name = _validate_collection_name(collection_name)
    doc_id = str(document_id).strip()
    if not doc_id:
        raise ValueError("Document id must be non-empty")

    existing = list_chroma_collections()
    if col_name not in existing:
        raise CollectionNotFoundError(f"Collection '{col_name}' not found")

    collection = _aquifer_chroma_db.get_collection(name=col_name, embedding_function=openai_ef)
    # Ensure documents are included in the response
    result = collection.get(ids=[doc_id])
    if not result["ids"]:
        raise DocumentNotFoundError(f"Document '{doc_id}' not found in collection '{col_name}'")
    documents = result.get("documents") or []
    if not documents:
        # Defensive: if API didn't return documents for some reason
        raise DocumentNotFoundError(f"Document '{doc_id}' has no text in collection '{col_name}'")
    return documents[0]


def list_document_ids_in_collection(name: str) -> list[str]:
    """Return all document ids for the given collection.

    Raises CollectionNotFoundError if the collection doesn't exist, and
    ValueError if the name is empty/invalid.
    """
    col_name = _validate_collection_name(name)
    existing = list_chroma_collections()
    if col_name not in existing:
        raise CollectionNotFoundError(f"Collection '{col_name}' not found")

    collection = _aquifer_chroma_db.get_collection(name=col_name, embedding_function=openai_ef)

    # Page through results to avoid huge single calls. Use a generous page size.
    page_size = 10000
    all_ids: list[str] = []
    offset = 0
    while True:
        # Some versions of chromadb support offset; if not, break after first page
        try:
            result = collection.get(limit=page_size, offset=offset)
        except TypeError:
            # Fallback for clients without offset support
            result = collection.get(limit=page_size)
        ids = result.get("ids") or []
        all_ids.extend(ids)
        if len(ids) < page_size:
            break
        offset += page_size
    return all_ids


def iter_collection_batches(
    collection: Any,
    *,
    batch_size: int = 1000,
    include_embeddings: bool = False,
) -> Iterator[dict[str, Any]]:
    """Yield batches of documents from a Chroma collection.

    Each yielded batch is a mapping resembling the structure returned by
    ``collection.get(...)`` with keys like "ids", "documents",
    "metadatas", and optionally "embeddings" when requested.

    We attempt to use offset-based pagination when available and fall back
    to a simple limit-only call otherwise.
    """
    kwargs: dict[str, Any] = {"limit": batch_size}
    # Chroma's include supports: ["documents", "embeddings", "metadatas",
    # "distances", "uris", "data"]. "ids" are always returned and must NOT
    # be included, or some servers raise.
    includes: list[str] = ["documents", "metadatas"]
    if include_embeddings:
        includes.append("embeddings")
    # Not all clients support include=..., so be defensive
    kwargs_with_include = dict(kwargs)
    kwargs_with_include["include"] = includes

    offset = 0
    while True:
        try:
            result = collection.get(offset=offset, **kwargs_with_include)
        except (TypeError, ValueError):
            # Fallback for clients without offset/include support or that
            # reject certain include values
            try:
                result = collection.get(**kwargs)
            except (TypeError, ValueError):
                # Last resort: no kwargs supported
                result = collection.get()
        ids: Sequence[str] = cast(Sequence[str], result.get("ids") or [])
        if not ids:
            break
        yield result
        if len(ids) < batch_size:
            break
        offset += batch_size


def max_numeric_id_in_collection(name: str) -> int:
    """Return the maximum numeric document id in the collection or 0 if none.

    Non-numeric ids are ignored. Raises CollectionNotFoundError if the
    collection is missing.
    """
    cleaned = _validate_collection_name(name)
    existing = list_chroma_collections()
    if cleaned not in existing:
        raise CollectionNotFoundError(f"Collection '{cleaned}' not found")
    collection = _aquifer_chroma_db.get_collection(name=cleaned, embedding_function=openai_ef)
    max_id = 0
    for batch in iter_collection_batches(collection, batch_size=10000, include_embeddings=False):
        ids = batch.get("ids") or []
        for _id in ids:
            if _id.isdigit():
                value = int(_id)
                max_id = max(max_id, value)
    return max_id


def get_chroma_collections_pair(source: str, dest: str) -> Tuple[Any, Any]:
    """Return (source_collection, dest_collection) if both exist, else raise."""
    src_name = _validate_collection_name(source)
    dst_name = _validate_collection_name(dest)
    existing = list_chroma_collections()
    if src_name not in existing:
        raise CollectionNotFoundError(f"Collection '{src_name}' not found")
    if dst_name not in existing:
        raise CollectionNotFoundError(f"Collection '{dst_name}' not found")
    source_col = _aquifer_chroma_db.get_collection(name=src_name, embedding_function=openai_ef)
    dest_col = _aquifer_chroma_db.get_collection(name=dst_name, embedding_function=openai_ef)
    return source_col, dest_col
