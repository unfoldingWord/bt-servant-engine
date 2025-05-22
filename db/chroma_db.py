import chromadb
from chromadb.utils import embedding_functions
from chromadb.api import ClientAPI
from config import Config
from logger import get_logger

DATA_DIR = Config.DATA_DIR
DATA_DIR.mkdir(parents=True, exist_ok=True)

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                model_name="text-embedding-ada-002",
                api_key=Config.OPENAI_API_KEY
            )
_aquifer_chroma_db = chromadb.PersistentClient(path=str(DATA_DIR))
_knowledge_base_collection = _aquifer_chroma_db.get_collection(name="knowledgebase", embedding_function=openai_ef)

logger = get_logger(__name__)


def get_chroma_collection(name) -> ClientAPI:
    return _aquifer_chroma_db.get_collection(name=name, embedding_function=openai_ef)


def add_knowledgebase_doc(doc: str) -> str:
    doc_id = str(get_next_doc_id(_knowledge_base_collection))
    logger.info("using next doc id: %s", doc_id)
    _knowledge_base_collection.upsert(
        documents=[doc],
        metadatas=[{"source": str("Knowledgebase User Override")}],
        ids=[str(4)],
    )
    logger.info("successfully inserted:\n\n%s\n\ninto the db.", doc)
    return doc_id


def get_next_doc_id(collection) -> int:
    results = collection.get(limit=10000)
    if not results["ids"]:
        return 1
    # Convert string IDs to integers and get the max
    int_ids = [int(doc_id) for doc_id in results["ids"] if doc_id.isdigit()]
    return max(int_ids, default=0) + 1
