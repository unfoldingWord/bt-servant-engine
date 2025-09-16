"""Intent classification node."""
# pylint: disable=duplicate-code
from __future__ import annotations

from typing import Any, cast

from openai.types.responses.easy_input_message_param import EasyInputMessageParam

from logger import get_logger
from servant_brain.classifier import (
    INTENT_CLASSIFICATION_AGENT_SYSTEM_PROMPT,
    UserIntents,
)
from servant_brain.dependencies import open_ai_client
from servant_brain.state import BrainState
from servant_brain.tokens import extract_cached_input_tokens
from utils.perf import add_tokens

logger = get_logger(__name__)


def determine_intents(state: Any) -> dict:
    """Classify the user's transformed query into one or more intents."""

    s = cast(BrainState, state)
    query = s["transformed_query"]
    messages: list[EasyInputMessageParam] = [
        {"role": "system", "content": INTENT_CLASSIFICATION_AGENT_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"what is your classification of the latest user message: {query}",
        },
    ]
    response = open_ai_client.responses.parse(
        model="gpt-4o",
        input=cast(Any, messages),
        text_format=UserIntents,
        store=False,
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
    user_intents_model = cast(UserIntents, response.output_parsed)
    logger.info(
        "extracted user intents: %s",
        " ".join(intent.value for intent in user_intents_model.intents),
    )
    return {"user_intents": user_intents_model.intents}


__all__ = ["determine_intents"]
