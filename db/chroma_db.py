import chromadb
from chromadb.utils import embedding_functions
from typing import Optional, Any
try:
    # Newer Chroma
    from chromadb.api.models.Collection import Collection as ChromaCollection  # type: ignore
except Exception:
    try:
        # Older alias in some versions
        from chromadb.api import Collection as ChromaCollection  # type: ignore
    except Exception:
        ChromaCollection = Any  # type: ignore
from config import config
from logger import get_logger

DATA_DIR = config.DATA_DIR
DATA_DIR.mkdir(parents=True, exist_ok=True)

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                model_name="text-embedding-ada-002",
                api_key=config.OPENAI_API_KEY
            )
_aquifer_chroma_db = chromadb.PersistentClient(path=str(DATA_DIR))

logger = get_logger(__name__)


def get_chroma_collection(name: str) -> Optional[ChromaCollection]:
    existing_collections = [col.name for col in _aquifer_chroma_db.list_collections()]
    if name in existing_collections:
        return _aquifer_chroma_db.get_collection(name=name, embedding_function=openai_ef)
    return None


def get_next_doc_id(collection: ChromaCollection) -> int:
    results = collection.get(limit=10000)
    if not results["ids"]:
        return 1
    # Convert string IDs to integers and get the max
    int_ids = [int(doc_id) for doc_id in results["ids"] if doc_id.isdigit()]
    return max(int_ids, default=0) + 1
