"""Vector database and RAG query infrastructure for BT Servant.

This module provides the final RAG response generation pipeline, including
querying vector databases (ChromaDB) and generating responses using OpenAI.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Optional, cast

from openai import OpenAI, OpenAIError
from openai.types.responses.easy_input_message_param import EasyInputMessageParam

from bt_servant_engine.core.intents import IntentType
from bt_servant_engine.core.logging import get_logger
from bt_servant_engine.services.openai_utils import track_openai_usage

logger = get_logger(__name__)

RELEVANCE_CUTOFF = 0.65
TOP_K = 5

FINAL_RESPONSE_AGENT_SYSTEM_PROMPT = """
You are an assistant to Bible translators. Your main job is to answer questions about content found in various biblical
resources: commentaries, translation notes, bible dictionaries, and various resources like FIA. In addition to answering
questions, you may be called upon to: summarize the data from resources, transform the data from resources (like
explaining it a 5-year old level, etc, and interact with the resources in all kinds of ways. All this is a part of your
responsibilities. Context from resources (RAG results) will be provided to help you answer the question(s). Only answer
questions using the provided context from resources!!! If you can't confidently figure it out using that context,
simply say 'Sorry, I couldn't find any information in my resources to service your request or command. But
maybe I'm unclear on your intent. Could you perhaps state it a different way?' You will also be given the past
conversation history. Use this to understand the user's current message or query if necessary. If the past conversation
history is not relevant to the user's current message, just ignore it. FINALLY, UNDER NO CIRCUMSTANCES ARE YOU TO SAY
ANYTHING THAT WOULD BE DEEMED EVEN REMOTELY HERETICAL BY ORTHODOX CHRISTIANS. If you can't do what the user is asking
because your response would be heretical, explain to the user why you cannot comply with their request or command.
"""


def query_vector_db(
    transformed_query: str,
    stack_rank_collections: list[str],
    get_chroma_collection_fn: Callable[[str], Any],
    boilerplate_features_message: str,
    *,
    top_k: int = TOP_K,
    relevance_cutoff: float = RELEVANCE_CUTOFF,
) -> dict[str, list[dict[str, str]]]:
    """Query the vector DB (Chroma) across ranked collections and filter by relevance.

    Args:
        transformed_query: The preprocessed user query
        stack_rank_collections: Collection names in priority order
        get_chroma_collection_fn: Function to retrieve a Chroma collection by name
        boilerplate_features_message: Fallback message when no docs are found
        top_k: Maximum number of documents to retrieve per collection
        relevance_cutoff: Minimum cosine similarity threshold for relevance

    Returns:
        Dictionary with "docs" key containing filtered document list
    """
    # pylint: disable=too-many-locals
    _ = boilerplate_features_message  # Not used in this function, but kept for signature
    filtered_docs = []
    # this loop is the current implementation of the "stacked ranked" algorithm
    for collection_name in stack_rank_collections:
        logger.info("querying stack collection: %s", collection_name)
        db_collection = get_chroma_collection_fn(collection_name)
        if not db_collection:
            logger.warning("collection %s was not found in chroma db.", collection_name)
            continue
        col = cast(Any, db_collection)
        results = col.query(query_texts=[transformed_query], n_results=top_k)
        docs = results["documents"]
        similarities = results["distances"]
        metadata = results["metadatas"]
        logger.info("\nquery: %s\n", transformed_query)
        logger.info("---")
        hits = 0
        for i in range(len(docs[0])):
            cosine_similarity = round(1 - similarities[0][i], 4)
            doc = docs[0][i]
            m = metadata[0][i]
            resource_name = m.get("name", "")
            source = m.get("source", "")
            logger.info("processing %s from %s.", resource_name, source)
            logger.info("Cosine Similarity: %s", cosine_similarity)
            logger.info("Metadata: %s", resource_name)
            logger.info("---")
            if cosine_similarity >= relevance_cutoff:
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


def query_open_ai(
    client: OpenAI,
    docs: list[dict[str, str]],
    transformed_query: str,
    chat_history: list[dict[str, str]],
    model_for_agentic_strength_fn: Callable[[str, bool, bool], str],
    extract_cached_input_tokens_fn: Callable[[Any], Optional[int]],
    add_tokens_fn: Callable[..., None],
    agentic_strength: str,
    boilerplate_features_message: str,
) -> dict[str, list[dict[str, Any]]]:
    """Generate the final response text using RAG context and OpenAI.

    Args:
        client: OpenAI client instance
        docs: Retrieved documents from vector DB
        transformed_query: Preprocessed user query
        chat_history: Conversation history
        model_for_agentic_strength_fn: Function to select model based on agentic strength
        extract_cached_input_tokens_fn: Function to extract cached token counts
        add_tokens_fn: Function to track token usage
        agentic_strength: User's agentic strength preference
        boilerplate_features_message: Fallback message when no docs are found

    Returns:
        Dictionary with "responses" key containing response list
    """
    # pylint: disable=too-many-locals
    try:
        if len(docs) == 0:
            no_docs_msg = (
                f"Sorry, I couldn't find any information in my resources to service your request "
                f"or command.\n\n{boilerplate_features_message}"
            )
            return {
                "responses": [
                    {
                        "intent": IntentType.GET_BIBLE_TRANSLATION_ASSISTANCE,
                        "response": no_docs_msg,
                    }
                ]
            }

        # build context from docs
        context = json.dumps(docs, indent=2)
        logger.info("context passed to final node:\n\n%s", context)
        rag_context_message = (
            "When answering my next query, use this additional" + f"  context: {context}"
        )
        chat_history_context_message = (
            f"Use this conversation history to understand the user's "
            f"current request only if needed: {json.dumps(chat_history)}"
        )
        messages = cast(
            list[EasyInputMessageParam],
            [
                {"role": "developer", "content": rag_context_message},
                {
                    "role": "developer",
                    "content": "Focus only on the portion of the user's message requesting general Bible translation assistance. Ignore unrelated requests or passages mentioned elsewhere in the message.",
                },
                {"role": "developer", "content": chat_history_context_message},
                {"role": "user", "content": transformed_query},
            ],
        )
        model_name = model_for_agentic_strength_fn(
            agentic_strength, allow_low=False, allow_very_low=True
        )
        response = client.responses.create(
            model=model_name,
            instructions=FINAL_RESPONSE_AGENT_SYSTEM_PROMPT,
            input=cast(Any, messages),
        )
        usage = getattr(response, "usage", None)
        track_openai_usage(usage, model_name, extract_cached_input_tokens_fn, add_tokens_fn)
        bt_servant_response = response.output_text
        logger.info("response from openai: %s", bt_servant_response)
        logger.debug("%d characters returned from openAI", len(bt_servant_response))

        resource_list = ", ".join(
            {
                f"{item.get('resource_name', 'unknown')} from {item.get('source', 'unknown')}"
                for item in docs
            }
        )
        cascade_info = (
            f"bt servant used the following resources to generate its response: {resource_list}."
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
            "I encountered some problems while trying to respond. Let Ian know about this one."
        )
        return {
            "responses": [
                {
                    "intent": IntentType.GET_BIBLE_TRANSLATION_ASSISTANCE,
                    "response": error_msg,
                }
            ]
        }
