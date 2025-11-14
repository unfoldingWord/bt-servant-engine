"""Intent handler for FIA (Familiarization, Internalization, Articulation) resources."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, List, Optional, cast

from openai import OpenAI, OpenAIError
from openai.types.responses.easy_input_message_param import EasyInputMessageParam

from bt_servant_engine.core.intents import IntentType
from bt_servant_engine.core.logging import get_logger
from bt_servant_engine.services.intents.simple_intents import (
    BOILER_PLATE_AVAILABLE_FEATURES_MESSAGE,
)
from bt_servant_engine.services.openai_utils import track_openai_usage
from utils.perf import add_tokens

logger = get_logger(__name__)

CONSULT_FIA_RESOURCES_SYSTEM_PROMPT = """
# Identity

You are the FIA specialist node of BT Servant. You help Bible translators understand and apply the Familiarization,
Internalization, and Articulation (FIA) process using only the supplied context.

# Context Handling

- You will always receive the official FIA reference document plus any retrieved FIA resource snippets.
- When the user's request is about the FIA process itself (for example, asking for the steps or how to translate the
  Bible in general), rely primarily on the FIA reference document. Quote or summarize the steps accurately and keep the
  sequence intact.
- When the user asks how FIA applies to a specific passage, language, or scenario, synthesize both the reference
  document and the retrieved snippets. Mention the relevant FIA steps explicitly (e.g., "Step 2: Setting the Stage").
- If the context does not contain the needed information, clearly say you cannot find it and invite the user to clarify.
- Never invent steps or procedures. Stay faithful to the provided materials.

# Response Style

- Be practical, encouraging, and concise while remaining thorough enough for translators to act on the guidance.
- Use natural paragraphs (no bullet lists unless the context itself is a list that must be echoed for clarity).
- Include references to FIA steps or resource names when they help the user follow along.
"""

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
FIA_REFERENCE_PATH = BASE_DIR / "sources" / "fia" / "fia.md"

RELEVANCE_CUTOFF = 0.65
TOP_K = 5

try:
    FIA_REFERENCE_CONTENT = FIA_REFERENCE_PATH.read_text(encoding="utf-8")
except FileNotFoundError:
    logger.warning("FIA reference file missing at %s", FIA_REFERENCE_PATH)
    FIA_REFERENCE_CONTENT = ""  # pylint: disable=invalid-name


@dataclass(slots=True)
class FIARequest:
    """Inputs required to consult FIA resources for a user query."""

    client: OpenAI
    query: str
    chat_history: list[dict[str, str]]
    user_response_language: Optional[str]
    query_language: Optional[str]
    agentic_strength: str


@dataclass(slots=True)
class FIADependencies:
    """External services needed to fulfill an FIA request."""

    get_chroma_collection: Callable[..., Any]
    model_for_agentic_strength: Callable[..., str]
    extract_cached_input_tokens: Callable[[Any], Optional[int]]


def consult_fia_resources(request: FIARequest, dependencies: FIADependencies) -> dict[str, Any]:
    """Answer FIA-specific questions using FIA collections and reference material."""
    candidate_lang = _resolve_candidate_language(request)

    vector_docs, collection_used = _gather_vector_documents(
        request.query,
        candidate_lang,
        dependencies,
    )

    context_docs = _build_context_documents(vector_docs)
    if not context_docs:
        return _fia_fallback()

    return _generate_fia_response(
        request,
        dependencies,
        context_docs,
        collection_used,
    )


def _resolve_candidate_language(request: FIARequest) -> str:
    candidate = (request.user_response_language or request.query_language or "en").strip().lower()
    return candidate or "en"


def _gather_vector_documents(
    query: str,
    candidate_lang: str,
    dependencies: FIADependencies,
) -> tuple[list[dict[str, str]], Optional[str]]:
    localized_collection = f"{candidate_lang}_fia_resources"
    logger.info("[consult-fia] primary collection candidate: %s", localized_collection)
    vector_docs = _query_collection(dependencies, query, localized_collection)
    if vector_docs:
        return vector_docs, localized_collection
    if localized_collection == "en_fia_resources":
        return [], None
    logger.info("[consult-fia] falling back to en_fia_resources collection")
    fallback_docs = _query_collection(dependencies, query, "en_fia_resources")
    if fallback_docs:
        return fallback_docs, "en_fia_resources"
    return [], None


def _query_collection(
    dependencies: FIADependencies,
    query: str,
    name: str,
) -> list[dict[str, str]]:
    collection = dependencies.get_chroma_collection(name)
    if not collection:
        logger.warning("[consult-fia] collection %s was not found in chroma db.", name)
        return []
    results = cast(Any, collection).query(query_texts=[query], n_results=TOP_K)
    document_rows = cast(list[list[str]], results.get("documents", []))
    if not document_rows:
        return []
    hits: list[dict[str, str]] = []
    doc_row = document_rows[0]
    distance_row = cast(
        list[float],
        (results.get("distances") or [[0.0] * len(doc_row)])[0],
    )
    metadata_row = cast(
        list[Any],
        (results.get("metadatas") or [[{} for _ in doc_row]])[0],
    )
    for idx, doc in enumerate(doc_row):
        similarity = _similarity_from_distances(distance_row, idx)
        metadata = metadata_row[idx] if idx < len(metadata_row) else {}
        metadata_dict = metadata if isinstance(metadata, dict) else {}
        logger.info(
            "[consult-fia] processing %s from %s with similarity %.4f",
            (metadata_dict.get("name", "") or "<unnamed>"),
            (metadata_dict.get("source", "") or "<unknown>"),
            similarity,
        )
        if similarity >= RELEVANCE_CUTOFF:
            hits.append(
                {
                    "collection_name": name,
                    "resource_name": cast(str, metadata_dict.get("name", "")),
                    "source": cast(str, metadata_dict.get("source", "")),
                    "document_text": cast(str, doc),
                }
            )
    if hits:
        logger.info("[consult-fia] found %d hit(s) in %s", len(hits), name)
    return hits


def _similarity_from_distances(distances: list[Any], idx: int) -> float:
    try:
        return 1 - float(distances[idx])
    except (IndexError, TypeError, ValueError):
        return 0.0


def _build_context_documents(vector_docs: list[dict[str, str]]) -> list[dict[str, str]]:
    docs: list[dict[str, str]] = []
    if FIA_REFERENCE_CONTENT:
        docs.append(
            {
                "collection_name": "fia_reference",
                "resource_name": "FIA Reference Manual",
                "source": str(FIA_REFERENCE_PATH),
                "document_text": FIA_REFERENCE_CONTENT,
            }
        )
    else:
        logger.warning("[consult-fia] FIA reference content unavailable")
    docs.extend(vector_docs)
    return docs


def _generate_fia_response(
    request: FIARequest,
    dependencies: FIADependencies,
    context_docs: list[dict[str, str]],
    collection_used: Optional[str],
) -> dict[str, Any]:
    """Call the FIA model with context and return the formatted response payload."""
    try:
        context_payload = json.dumps(context_docs, indent=2)
        logger.info("[consult-fia] context passed to LLM:\n%s", context_payload)
        messages = _build_messages(request.query, request.chat_history, context_payload)
        fia_response = _call_fia_model(request, dependencies, messages)
    except OpenAIError:
        logger.error("[consult-fia] Error during OpenAI request", exc_info=True)
        error_msg = (
            "I encountered some problems while consulting FIA resources. "
            "Please let Ian know about this one."
        )
        return {"responses": [{"intent": IntentType.CONSULT_FIA_RESOURCES, "response": error_msg}]}

    _log_vector_usage(context_docs)
    update: dict[str, Any] = {
        "responses": [{"intent": IntentType.CONSULT_FIA_RESOURCES, "response": fia_response}]
    }
    if collection_used:
        update["collection_used"] = collection_used
    return update


def _fia_fallback() -> dict[str, Any]:
    fallback = (
        "Sorry, I couldn't find any FIA resources to service your request or command.\n\n"
        f"{BOILER_PLATE_AVAILABLE_FEATURES_MESSAGE}"
    )
    return {"responses": [{"intent": IntentType.CONSULT_FIA_RESOURCES, "response": fallback}]}


def _build_messages(
    query: str,
    chat_history: list[dict[str, str]],
    context_payload: str,
) -> List[EasyInputMessageParam]:
    return cast(
        List[EasyInputMessageParam],
        [
            {
                "role": "developer",
                "content": f"FIA context resources: {context_payload}",
            },
            {
                "role": "developer",
                "content": (
                    "Focus only on the portion of the user's message that requests FIA guidance. "
                    "Ignore any other requests or book references in the message."
                ),
            },
            {
                "role": "developer",
                "content": f"Use this conversation history if helpful: {json.dumps(chat_history)}",
            },
            {
                "role": "user",
                "content": query,
            },
        ],
    )


def _call_fia_model(
    request: FIARequest,
    dependencies: FIADependencies,
    messages: list[EasyInputMessageParam],
) -> str:
    model_name = dependencies.model_for_agentic_strength(
        request.agentic_strength,
        allow_low=True,
        allow_very_low=True,
    )
    response = request.client.responses.create(
        model=model_name,
        instructions=CONSULT_FIA_RESOURCES_SYSTEM_PROMPT,
        input=cast(Any, messages),
    )
    usage = getattr(response, "usage", None)
    track_openai_usage(
        usage,
        model_name,
        dependencies.extract_cached_input_tokens,
        add_tokens,
    )
    fia_response = response.output_text
    logger.info("[consult-fia] response from openai: %s", fia_response)
    return fia_response


def _log_vector_usage(context_docs: list[dict[str, str]]) -> None:
    resource_list = ", ".join(
        {
            (f"{doc.get('resource_name', 'unknown')} from {doc.get('source', 'unknown')}")
            for doc in context_docs
            if doc.get("collection_name") != "fia_reference"
        }
    )
    if resource_list:
        logger.info("[consult-fia] vector resources used: %s", resource_list)


__all__ = [
    "CONSULT_FIA_RESOURCES_SYSTEM_PROMPT",
    "consult_fia_resources",
]
