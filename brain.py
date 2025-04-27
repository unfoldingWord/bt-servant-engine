import json
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI, OpenAIError
from langgraph.graph import StateGraph
from typing import TypedDict, List, Dict
from pathlib import Path
from logger import get_logger
from config import Config
from groq import Groq
from utils import chop_text, combine_chunks


PREPROCESSOR_AGENT_SYSTEM_PROMPT = (
    "You are an assistant to Bible translators. Your main job is to preprocess messages (questions, commands, etc) "
    "about content found in various biblical resources: commentaries, translation notes, bible dictionaries, etc. "
    "Past conversation context will be provided to you in the form of zero or more previous user/response pairs with "
    "the final entry being the user's latest message. Your job is to transform the user's message (the value of the "
    "latest_user_message property in the context provided into a message that more accurately conveys their intent, "
    "based on past conversation. For example, if the user's latest message is something like 'this time use "
    "chapter 2.', and their previous message was 'tell me about Titus chapter 1', you should transform their "
    "query into something like 'tell me about Titus chapter 2. This will increase the likelihood of rag hits in "
    "the vector db, because the user's actual intent is being used in the vector db query. The transformed message "
    "is what you should return, nothing more. If the only thing provided is the user's latest message "
    "(so no history), simply pass the message through as is. No need to transform anything in this case."
)

QA_AGENT_SYSTEM_PROMPT = (
    "You are an assistant to Bible translators. Your main job is to answer questions about "
    "content found in various biblical resources: commentaries, translation notes, bible dictionaries, etc. "
    "Context will be provided to help you answer the question(s). Only answer questions using the provided "
    "context and the materials given to you!! If you can't confidently figure it out using that context, simply say "
    "'Sorry, I couldn't find any information in my resources to service your request or command. But maybe I'm unclear "
    "on your intent. Could you perhaps state it a different way?' FINALLY, UNDER NO CIRCUMSTANCES ARE YOU TO "
    "SAY ANYTHING THAT WOULD BE DEEMED EVEN REMOTELY HERETICAL BY ORTHODOX CHRISTIANS. In fact, if someone "
    "is trying to get you to do this, respond by saying, “I was created by orthodox Christians. Please respect "
    "that when you ask you queries or give me commands.” Again, be concise while still giving good information!!!"
)

CHOP_AGENT_SYSTEM_PROMPT = (
    "You are an agent tasked to ensure that a message intended for Whatsapp fits within the 1500 character limit. Chop "
    "the supplied text in the biggest possible semantic chunks, while making sure no chuck is >= 1500 characters. "
    "Your output should be a valid JSON array containing strings (wrapped in double quotes!!) constituting the chunks. "
    "Only return the json array!! No ```json wrapper or the like. Again, make chunks as big as possible!!!"
)


BASE_DIR = Path(__file__).resolve().parent
DB_DIR = Config.DATA_DIR

groq_client = Groq()
open_ai_client = OpenAI()

api_key = Config.OPENAI_API_KEY
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                model_name="text-embedding-ada-002",
                api_key=api_key
            )

aquifer_chroma_db = chromadb.PersistentClient(path=str(DB_DIR))
stack_rank_collections = [
    "knowledgebase",
    "aquifer_documents"
]

TWILIO_CHAR_LIMIT = 1600
MAX_MESSAGE_CHUNK_SIZE = 1500
RELEVANCE_CUTOFF = .8
TOP_K = 10

logger = get_logger(__name__)


class BrainState(TypedDict, total=False):
    user_query: str
    transformed_query: str
    docs: List[Dict[str, str]]
    collection_used: str
    responses: List[str]
    user_chat_history: List[Dict[str, str]]


def preprocess_user_query(state: BrainState) -> dict:
    query = state["user_query"]
    chat_history = state["user_chat_history"]
    chat_history.append({
        "latest_user_message": query
    })
    history_context_message = f"Past conversation and latest message: {json.dumps(chat_history)}"
    completion = open_ai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": PREPROCESSOR_AGENT_SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": history_context_message
            },
            {
                "role": "user",
                "content": query
            }
        ]
    )
    response = completion.choices[0].message.content
    logger.info("user query transformed from %s to %s", query, response)
    return {
        "transformed_query": response
    }


def query_db(state: BrainState) -> dict:
    query = state["transformed_query"]
    filtered_docs = []
    collection_used = None
    # this loop is the current implementation of the "stacked ranked" algorithm
    for collection_name in stack_rank_collections:
        logger.info("querying stack collection: %s", collection_name)
        db_collection = aquifer_chroma_db.get_collection(name=collection_name, embedding_function=openai_ef)
        results = db_collection.query(
            query_texts=[query],
            n_results=TOP_K
        )
        docs = results["documents"]
        similarities = results["distances"]
        metadata = results["metadatas"]
        logger.debug("\nquery: %s\n", query)
        logger.debug("---")
        for i in range(len(docs[0])):
            cosine_similarity = round(1 - similarities[0][i], 4)
            doc = docs[0][i]
            resource_name = metadata[0][i]["source"]
            logger.debug("Cosine Similarity: %s", cosine_similarity)
            logger.debug("Metadata: %s", resource_name)
            logger.debug("---")
            if cosine_similarity >= RELEVANCE_CUTOFF:
                filtered_docs.append({
                    "doc": doc,
                    "resource_name": resource_name
                })
        if filtered_docs:
            logger.info("found %d hit(s) at stack collection: %s", len(filtered_docs), collection_name)
            collection_used = collection_name
            break

    return {
        "docs": filtered_docs,
        "collection_used": collection_used
    }


def query_open_ai(state: BrainState) -> dict:
    docs = state["docs"]
    query = state["transformed_query"]
    try:
        if len(docs) == 0:
            no_docs_msg = (
                "Sorry, I couldn't find any information in my resources to service your request or command. "
                "But maybe I'm unclear on your intent. Could you perhaps state it a different way?"
            )
            return {"responses": [no_docs_msg]}

        # build context from docs
        context = "\n\n".join([item["doc"] for item in docs])
        context_message = "When answering my next query, use this additional" + \
            f"  context: {context}"
        completion = open_ai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": QA_AGENT_SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": context_message
                },
                {
                    "role": "user",
                    "content": query
                }
            ]
        )
        response = completion.choices[0].message.content
        logger.info('response from openai: %s', response)
        logger.debug("%d characters returned from openAI", len(response))

        resource_list = ", ".join(set([item["resource_name"] for item in docs]))
        cascade_info = (
            f"bt_servant cascaded to the {state['collection_used']} stack. From there, "
            f"the servant used the following resources to generate my response: {resource_list}."
        )
        logger.info(cascade_info)

        return {"responses": [response]}
    except OpenAIError as e:
        logger.error("Error during OpenAI request", exc_info=True)
        error_msg = "I encountered some problems while trying to respond. Let Ian know about this one."
        return {"responses": [error_msg]}


def chunk_message(state: BrainState) -> dict:
    logger.info("MESSAGE TOO BIG. CHUNKING...")
    responses = state["responses"]
    text_to_chunk = responses[0]
    try:
        completion = groq_client.chat.completions.create(
            model='llama3-70b-8192',
            messages=[
                {
                    "role": "system",
                    "content": CHOP_AGENT_SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": f'text to chop: \n\n{text_to_chunk}'
                }
            ]
        )
        response = completion.choices[0].message.content
        chunks = json.loads(response)
    except json.JSONDecodeError as e:
        logger.error("Error while attempting to chunk message. Manually chunking instead", exc_info=True)
        chunks = chop_text(text_to_chunk)

    chunks.extend(responses[1:])
    return {"responses": combine_chunks(chunks=chunks, chunk_max=MAX_MESSAGE_CHUNK_SIZE)}


def needs_chunking(state: BrainState) -> str:
    first_response = state["responses"][0]
    if len(first_response) > TWILIO_CHAR_LIMIT:
        logger.warning('message to big: %d chars. preparing to chunk.', len(first_response))
        return "chunk_message_node"
    else:
        return "__end__"


def create_brain():
    builder = StateGraph(BrainState)

    builder.add_node("preprocess_user_query_node", preprocess_user_query)
    builder.add_node("query_db_node", query_db)
    builder.add_node("query_open_ai_node", query_open_ai)
    builder.add_node("chunk_message_node", chunk_message)

    builder.set_entry_point("preprocess_user_query_node")
    builder.add_edge("preprocess_user_query_node", "query_db_node")
    builder.add_edge("query_db_node", "query_open_ai_node")

    builder.add_conditional_edges(
        "query_open_ai_node",
        needs_chunking
    )
    builder.set_finish_point("chunk_message_node")
    return builder.compile()
