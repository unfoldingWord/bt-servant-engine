"""Vector retrieval and final response generation nodes."""
# pylint: disable=duplicate-code
from __future__ import annotations

import json
from typing import Any, List, cast

from openai import OpenAIError
from openai.types.responses.easy_input_message_param import EasyInputMessageParam

from logger import get_logger
from servant_brain.classifier import IntentType
from servant_brain.dependencies import RELEVANCE_CUTOFF, TOP_K, open_ai_client
from servant_brain.nodes.capabilities import BOILER_PLATE_AVAILABLE_FEATURES_MESSAGE
from servant_brain.prompts import FINAL_RESPONSE_AGENT_SYSTEM_PROMPT
from servant_brain.state import BrainState
from servant_brain.tokens import extract_cached_input_tokens
from utils.perf import add_tokens
from db import get_chroma_collection

logger = get_logger(__name__)


def query_vector_db(state: Any) -> dict:
    """Query the vector DB (Chroma) across ranked collections and filter by relevance."""

    s = cast(BrainState, state)
    query = s["transformed_query"]
    stack_rank_collections = s["stack_rank_collections"]
    filtered_docs = []
    for collection_name in stack_rank_collections:
        logger.info("querying stack collection: %s", collection_name)
        db_collection = get_chroma_collection(collection_name)
        if not db_collection:
            logger.warning("collection %s was not found in chroma db.", collection_name)
            continue
        col = cast(Any, db_collection)
        results = col.query(query_texts=[query], n_results=TOP_K)
        docs = results["documents"]
        similarities = results["distances"]
        metadata = results["metadatas"]
        logger.info("\nquery: %s\n", query)
        logger.info("---")
        hits = 0
        for i in range(len(docs[0])):
            cosine_similarity = round(1 - similarities[0][i], 4)
            doc = docs[0][i]
            meta = metadata[0][i]
            resource_name = meta.get("name", "")
            source = meta.get("source", "")
            logger.info("processing %s from %s.", resource_name, source)
            logger.info("Cosine Similarity: %s", cosine_similarity)
            logger.info("Metadata: %s", resource_name)
            logger.info("---")
            if cosine_similarity >= RELEVANCE_CUTOFF:
                hits += 1
                filtered_docs.append(
                    {
                        "collection_name": collection_name,
                        "resource_name": resource_name,
                        "source": source,
                        "document_text": doc,
                    }
                )
        if hits > 0:
            logger.info("found %d hit(s) at stack collection: %s", hits, collection_name)

    return {"docs": filtered_docs}


# pylint: disable=too-many-locals

def query_open_ai(state: Any) -> dict:
    """Generate the final response text using RAG context and OpenAI."""

    s = cast(BrainState, state)
    docs = s["docs"]
    query = s["transformed_query"]
    chat_history = s["user_chat_history"]
    try:
        if len(docs) == 0:
            no_docs_msg = (
                "Sorry, I couldn't find any information in my resources to service your request "
                "or command.\n\n"
                f"{BOILER_PLATE_AVAILABLE_FEATURES_MESSAGE}"
            )
            return {
                "responses": [
                    {
                        "intent": IntentType.GET_BIBLE_TRANSLATION_ASSISTANCE,
                        "response": no_docs_msg,
                    },
                ]
            }

        context = json.dumps(docs, indent=2)
        logger.info("context passed to final node:\n\n%s", context)
        rag_context_message = (
            "When answering my next query, use this additional  context: "
            f"{context}"
        )
        chat_history_context_message = (
            "Use this conversation history to understand the user's current request "
            "only if needed: "
            f"{json.dumps(chat_history)}"
        )
        messages = cast(List[EasyInputMessageParam], [
            {"role": "developer", "content": rag_context_message},
            {"role": "developer", "content": chat_history_context_message},
            {"role": "user", "content": query},
        ])
        response = open_ai_client.responses.create(
            model="gpt-4o",
            instructions=FINAL_RESPONSE_AGENT_SYSTEM_PROMPT,
            input=cast(Any, messages),
        )
        usage = getattr(response, "usage", None)
        if usage is not None:
            it = getattr(usage, "input_tokens", None)
            ot = getattr(usage, "output_tokens", None)
            tt = getattr(usage, "total_tokens", None)
            if tt is None and (it is not None or ot is not None):
                tt = (it or 0) + (ot or 0)
            cit = extract_cached_input_tokens(usage)
            add_tokens(it, ot, tt, model="gpt-4o", cached_input_tokens=cit)
        bt_servant_response = response.output_text
        logger.info('response from openai: %s', bt_servant_response)
        logger.debug("%d characters returned from openAI", len(bt_servant_response))

        resource_list = ", ".join(
            {
                (
                    f"{item.get('resource_name', 'unknown')} "
                    f"from {item.get('source', 'unknown')}"
                )
                for item in docs
            }
        )
        cascade_info = (
            "bt servant used the following resources to generate its response: "
            f"{resource_list}."
        )
        logger.info(cascade_info)

        return {
            "responses": [
                {
                    "intent": IntentType.GET_BIBLE_TRANSLATION_ASSISTANCE,
                    "response": bt_servant_response,
                }
            ]
        }
    except OpenAIError:
        logger.error("Error during OpenAI request", exc_info=True)
        error_msg = (
            "I encountered some problems while trying to respond. "
            "Let Ian know about this one."
        )
        return {
            "responses": [
                {"intent": IntentType.GET_BIBLE_TRANSLATION_ASSISTANCE, "response": error_msg},
            ]
        }


__all__ = ["query_vector_db", "query_open_ai"]
