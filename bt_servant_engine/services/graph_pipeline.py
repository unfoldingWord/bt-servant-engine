"""Vector database and RAG query infrastructure for BT Servant.

This module provides the final RAG response generation pipeline, including
querying vector databases (ChromaDB) and generating responses using OpenAI.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
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


@dataclass(slots=True)
class OpenAIQueryDependencies:
    """Container for helpers needed when building OpenAI responses."""

    model_for_agentic_strength: Callable[..., str]
    extract_cached_input_tokens: Callable[[Any], Optional[int]]
    add_tokens: Callable[..., None]


@dataclass(slots=True)
class VectorQueryConfig:
    """Configuration for vector search limits and relevance thresholds."""

    top_k: int = TOP_K
    relevance_cutoff: float = RELEVANCE_CUTOFF


@dataclass(slots=True)
class OpenAIQueryPayload:
    """Inputs required for the final OpenAI response generation."""

    docs: list[dict[str, Any]]
    transformed_query: str
    chat_history: list[dict[str, str]]
    agentic_strength: str
    boilerplate_features_message: str


def _extract_query_rows(results: Any) -> tuple[list[str], list[float], list[Any]]:
    document_rows = cast(list[list[str]], results.get("documents", []))
    if not document_rows:
        return [], [], []
    document_row = document_rows[0]
    distance_row = cast(
        list[float],
        (results.get("distances") or [[0.0] * len(document_row)])[0],
    )
    metadata_row = cast(
        list[Any],
        (results.get("metadatas") or [[{} for _ in document_row]])[0],
    )
    return document_row, distance_row, metadata_row


def _query_collection_docs(
    transformed_query: str,
    collection_name: str,
    get_chroma_collection_fn: Callable[[str], Any],
    vector_config: VectorQueryConfig,
) -> list[dict[str, Any]]:
    """Return filtered documents for a single collection."""
    logger.info("querying stack collection: %s", collection_name)
    db_collection = get_chroma_collection_fn(collection_name)
    if not db_collection:
        logger.warning("collection %s was not found in chroma db.", collection_name)
        return []

    results = cast(Any, db_collection).query(
        query_texts=[transformed_query],
        n_results=vector_config.top_k,
    )

    document_row, distance_row, metadata_row = _extract_query_rows(results)
    if not document_row:
        return []

    logger.info("\nquery: %s\n", transformed_query)
    logger.info("---")

    filtered: list[dict[str, Any]] = []
    for document, distance_value, metadata_item in zip(
        document_row, distance_row, metadata_row, strict=False
    ):
        cosine_similarity = round(1 - distance_value, 4)
        metadata_entry = (
            dict(cast(dict[str, Any], metadata_item)) if isinstance(metadata_item, dict) else {}
        )
        logger.info(
            "processing %s from %s.",
            metadata_entry.get("name", ""),
            metadata_entry.get("source", ""),
        )
        logger.info("Cosine Similarity: %s", cosine_similarity)
        logger.info("Metadata: %s", metadata_entry.get("name", ""))
        logger.info("---")
        if cosine_similarity < vector_config.relevance_cutoff:
            continue
        filtered.append(
            {
                "collection_name": collection_name,
                "resource_name": metadata_entry.get("name", ""),
                "source": metadata_entry.get("source", ""),
                "document_text": document,
                "metadata": metadata_entry,
            }
        )

    if filtered:
        logger.info("found %d hit(s) at stack collection: %s", len(filtered), collection_name)
    return filtered


def _no_docs_response(boilerplate_features_message: str) -> dict[str, list[dict[str, Any]]]:
    """Return response payload when no documents are retrieved."""
    message = (
        "Sorry, I couldn't find any information in my resources to service your request "
        "or command.\n\n"
        f"{boilerplate_features_message}"
    )
    return {
        "responses": [
            {
                "intent": IntentType.GET_BIBLE_TRANSLATION_ASSISTANCE,
                "response": message,
            }
        ]
    }


def _build_rag_messages(
    payload: OpenAIQueryPayload,
    context: str,
) -> list[EasyInputMessageParam]:
    """Create the developer/user messages for the OpenAI Responses API call."""
    rag_context_message = f"When answering my next query, use this additional context: {context}"
    chat_history_json = json.dumps(payload.chat_history)
    chat_history_context_message = (
        "Use this conversation history to understand the user's current request "
        f"only if needed: {chat_history_json}"
    )
    return cast(
        list[EasyInputMessageParam],
        [
            {"role": "developer", "content": rag_context_message},
            {
                "role": "developer",
                "content": (
                    "Focus only on the portion of the user's message requesting general "
                    "Bible translation assistance. Ignore unrelated requests or passages "
                    "mentioned elsewhere in the message."
                ),
            },
            {"role": "developer", "content": chat_history_context_message},
            {"role": "user", "content": payload.transformed_query},
        ],
    )


def _summarize_resource_usage(docs: list[dict[str, Any]]) -> str:
    """Return a comma-separated list of resource names referenced in the response."""
    resources = {
        f"{item.get('resource_name', 'unknown')} from {item.get('source', 'unknown')}"
        for item in docs
    }
    return ", ".join(sorted(resources))


def query_vector_db(
    transformed_query: str,
    stack_rank_collections: list[str],
    get_chroma_collection_fn: Callable[[str], Any],
    _boilerplate_features_message: str,
    *,
    config: VectorQueryConfig | None = None,
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
    vector_config = config or VectorQueryConfig()
    filtered_docs: list[dict[str, Any]] = []
    # this loop is the current implementation of the "stacked ranked" algorithm
    for collection_name in stack_rank_collections:
        filtered_docs.extend(
            _query_collection_docs(
                transformed_query,
                collection_name,
                get_chroma_collection_fn,
                vector_config,
            )
        )
    return {"docs": filtered_docs}


def query_open_ai(
    client: OpenAI,
    payload: OpenAIQueryPayload,
    dependencies: OpenAIQueryDependencies,
) -> dict[str, list[dict[str, Any]]]:
    """Generate the final response text using RAG context and OpenAI.

    Args:
        client: OpenAI client instance
        payload: Combined RAG context, chat history, agentic strength, and fallback text
        dependencies: Container for model selection and token tracking helpers

    Returns:
        Dictionary with "responses" key containing response list
    """
    try:
        if not payload.docs:
            return _no_docs_response(payload.boilerplate_features_message)

        context = json.dumps(payload.docs, indent=2)
        logger.info("context passed to final node:\n\n%s", context)
        messages = _build_rag_messages(payload, context)
        model_name = dependencies.model_for_agentic_strength(
            payload.agentic_strength,
            allow_low=False,
            allow_very_low=True,
        )
        response = client.responses.create(
            model=model_name,
            instructions=FINAL_RESPONSE_AGENT_SYSTEM_PROMPT,
            input=cast(Any, messages),
        )
        usage = getattr(response, "usage", None)
        track_openai_usage(
            usage,
            model_name,
            dependencies.extract_cached_input_tokens,
            dependencies.add_tokens,
        )
        bt_servant_response = response.output_text
        logger.info("response from openai: %s", bt_servant_response)
        logger.debug("%d characters returned from openAI", len(bt_servant_response))

        resource_summary = _summarize_resource_usage(payload.docs)
        logger.info(
            "bt servant used the following resources to generate its response: %s",
            resource_summary,
        )

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
