"""Intent handler for FIA (Familiarization, Internalization, Articulation) resources."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, List, Optional, cast

from openai import OpenAI, OpenAIError
from openai.types.responses.easy_input_message_param import EasyInputMessageParam

from bt_servant_engine.core.intents import IntentType
from bt_servant_engine.core.language import SUPPORTED_LANGUAGE_MAP as supported_language_map
from bt_servant_engine.core.logging import get_logger
from bt_servant_engine.services.intents.simple_intents import (
    BOILER_PLATE_AVAILABLE_FEATURES_MESSAGE,
)
from bt_servant_engine.services.openai_utils import track_openai_usage
from utils.perf import add_tokens

logger = get_logger(__name__)

CONSULT_FIA_RESOURCES_SYSTEM_PROMPT_BASE = """
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

CONSULT_FIA_RESOURCES_FOLLOWUP_GUIDANCE = """
# Follow-up Question

After providing your response, suggest a related follow-up question that flows naturally from the current topic. Leverage
the user's original query and the FIA steps you discussed to make it specific and helpful. For example:
- If you explained Step 2 (Setting the Stage), suggest exploring Step 3 or how to apply Step 2 to their language
- If they asked about a specific passage, suggest applying another FIA step to that passage
- If they asked about the FIA process generally, suggest exploring how to apply it to a real translation scenario

Generate this naturally in the response language and make it contextual to what you just discussed (e.g., "Would you
like to explore how to apply Step 3 to this passage?" or "Should I explain how the Internalization phase works for your
target language?").

Strongly prefer an open-ended follow-up that starts with "What", "Which", "How", or "Where" so the user has to name the
next focus. If you must use a yes/no framing, make it highly specific so a simple "yes" still captures the exact action
the system should pursue (e.g., "Would you like me to outline how Step 3 addresses checking draft verses with your
language community next?"). Avoid generic yes/no prompts like "Would you like more help with FIA?".
"""

# Backward compatibility constant (defaults to including follow-up guidance).
CONSULT_FIA_RESOURCES_SYSTEM_PROMPT = (
    CONSULT_FIA_RESOURCES_SYSTEM_PROMPT_BASE + CONSULT_FIA_RESOURCES_FOLLOWUP_GUIDANCE
)

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
FIA_REFERENCE_PATH = BASE_DIR / "sources" / "fia" / "fia.md"

RELEVANCE_CUTOFF = 0.65
TOP_K = 5

try:
    FIA_REFERENCE_CONTENT = FIA_REFERENCE_PATH.read_text(encoding="utf-8")
except FileNotFoundError:
    logger.warning("FIA reference file missing at %s", FIA_REFERENCE_PATH)
    FIA_REFERENCE_CONTENT = ""  # pylint: disable=invalid-name


def consult_fia_resources(
    client: OpenAI,
    query: str,
    chat_history: list[dict[str, str]],
    user_response_language: Optional[str],
    query_language: Optional[str],
    get_chroma_collection_fn: Callable[..., Any],
    model_for_agentic_strength_fn: Callable[..., Any],
    extract_cached_input_tokens_fn: Callable[..., Any],
    agentic_strength: str,
    *,
    include_followup: bool = True,
    ignored_topics: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Answer FIA-specific questions using FIA collections and reference material."""
    candidate_lang = (user_response_language or query_language or "en").lower()
    if candidate_lang not in supported_language_map:
        candidate_lang = "en"

    localized_collection = f"{candidate_lang}_fia_resources"
    logger.info("[consult-fia] primary collection candidate: %s", localized_collection)

    def _query_collection(name: str) -> list[dict[str, str]]:
        collection = get_chroma_collection_fn(name)
        if not collection:
            logger.warning("[consult-fia] collection %s was not found in chroma db.", name)
            return []
        chroma_collection = cast(Any, collection)
        results = chroma_collection.query(query_texts=[query], n_results=TOP_K)
        documents = cast(list, results.get("documents", []))
        distances = cast(list, results.get("distances", []))
        metadatas = cast(list, results.get("metadatas", []))
        if not documents:
            return []
        hits: list[dict[str, str]] = []
        docs_for_query = documents[0]
        dists_for_query = distances[0] if distances else []
        metas_for_query = metadatas[0] if metadatas else []
        for idx, doc in enumerate(docs_for_query):
            try:
                similarity = 1 - float(dists_for_query[idx])
            except (IndexError, TypeError, ValueError):
                similarity = 0.0
            metadata = metas_for_query[idx] if idx < len(metas_for_query) else {}
            resource_name = (
                cast(str, metadata.get("name", "")) if isinstance(metadata, dict) else ""
            )
            source = cast(str, metadata.get("source", "")) if isinstance(metadata, dict) else ""
            logger.info(
                "[consult-fia] processing %s from %s with similarity %.4f",
                resource_name or "<unnamed>",
                source or "<unknown>",
                similarity,
            )
            if similarity >= RELEVANCE_CUTOFF:
                hits.append(
                    {
                        "collection_name": name,
                        "resource_name": resource_name,
                        "source": source,
                        "document_text": cast(str, doc),
                    }
                )
        if hits:
            logger.info("[consult-fia] found %d hit(s) in %s", len(hits), name)
        return hits

    vector_docs = _query_collection(localized_collection)
    collection_used: Optional[str] = localized_collection if vector_docs else None
    if not vector_docs and localized_collection != "en_fia_resources":
        logger.info("[consult-fia] falling back to en_fia_resources collection")
        vector_docs = _query_collection("en_fia_resources")
        if vector_docs:
            collection_used = "en_fia_resources"

    context_docs: list[dict[str, str]] = []
    if FIA_REFERENCE_CONTENT:
        context_docs.append(
            {
                "collection_name": "fia_reference",
                "resource_name": "FIA Reference Manual",
                "source": str(FIA_REFERENCE_PATH),
                "document_text": FIA_REFERENCE_CONTENT,
            }
        )
    else:
        logger.warning("[consult-fia] FIA reference content unavailable")

    context_docs.extend(vector_docs)

    if not context_docs:
        fallback = (
            "Sorry, I couldn't find any FIA resources to service your request or command.\n\n"
            f"{BOILER_PLATE_AVAILABLE_FEATURES_MESSAGE}"
        )
        return {"responses": [{"intent": IntentType.CONSULT_FIA_RESOURCES, "response": fallback}]}

    context_payload = json.dumps(context_docs, indent=2)
    logger.info("[consult-fia] context passed to LLM:\n%s", context_payload)

    ignore_list = ignored_topics or []

    messages = cast(
        List[EasyInputMessageParam],
        [
            {
                "role": "developer",
                "content": f"FIA context resources: {context_payload}",
            },
            {
                "role": "developer",
                "content": "Focus only on the portion of the user's message that requests FIA guidance. Ignore any other requests or book references in the message.",
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
    if ignore_list:
        messages.insert(
            2,
            {
                "role": "developer",
                "content": (
                    "The user has other pending requests that will be handled later. Do NOT address these topics now: "
                    + ", ".join(ignore_list)
                    + "."
                ),
            },
        )
    if not include_followup:
        messages.append(
            {
                "role": "developer",
                "content": "Do not add a follow-up question. End the response after covering the requested FIA guidance.",
            }
        )

    try:
        model_name = model_for_agentic_strength_fn(
            agentic_strength, allow_low=True, allow_very_low=True
        )
        response = client.responses.create(
            model=model_name,
            instructions=(
                CONSULT_FIA_RESOURCES_SYSTEM_PROMPT_BASE + CONSULT_FIA_RESOURCES_FOLLOWUP_GUIDANCE
                if include_followup
                else CONSULT_FIA_RESOURCES_SYSTEM_PROMPT_BASE
            ),
            input=cast(Any, messages),
        )
        usage = getattr(response, "usage", None)
        track_openai_usage(usage, model_name, extract_cached_input_tokens_fn, add_tokens)

        fia_response = response.output_text
        logger.info("[consult-fia] response from openai: %s", fia_response)

        resource_list = ", ".join(
            {
                (f"{doc.get('resource_name', 'unknown')} from {doc.get('source', 'unknown')}")
                for doc in context_docs
                if doc.get("collection_name") != "fia_reference"
            }
        )
        if resource_list:
            logger.info("[consult-fia] vector resources used: %s", resource_list)

        update: dict[str, Any] = {
            "responses": [{"intent": IntentType.CONSULT_FIA_RESOURCES, "response": fia_response}]
        }
        if collection_used:
            update["collection_used"] = collection_used
        return update
    except OpenAIError:
        logger.error("[consult-fia] Error during OpenAI request", exc_info=True)
        error_msg = "I encountered some problems while consulting FIA resources. Please let Ian know about this one."
        return {"responses": [{"intent": IntentType.CONSULT_FIA_RESOURCES, "response": error_msg}]}


__all__ = [
    "CONSULT_FIA_RESOURCES_SYSTEM_PROMPT_BASE",
    "CONSULT_FIA_RESOURCES_FOLLOWUP_GUIDANCE",
    "CONSULT_FIA_RESOURCES_SYSTEM_PROMPT",
    "consult_fia_resources",
]
