"""Preprocess user query node."""
# pylint: disable=duplicate-code
from __future__ import annotations

import json
from typing import Any, cast

from openai.types.responses.easy_input_message_param import EasyInputMessageParam
from pydantic import BaseModel

from logger import get_logger
from servant_brain.dependencies import open_ai_client
from servant_brain.state import BrainState
from servant_brain.tokens import extract_cached_input_tokens
from utils.perf import add_tokens

logger = get_logger(__name__)


PREPROCESSOR_AGENT_SYSTEM_PROMPT = """
# Identity

You are a preprocessor agent/node in a retrieval augmented generation (RAG) pipeline. 

# Instructions

Use past conversation context, 
if supplied and applicable, to disambiguate or clarify the intent or meaning of the user's current message. Change 
as little as possible. Change nothing unless necessary. If the intent of the user's message is already clear, 
change nothing. Never greatly expand the user's current message. Changes should be small or none. Feel free to fix 
obvious spelling mistakes or errors, but not logic errors like incorrect books of the Bible. Do NOT narrow the scope of
explicit scripture selections: if a user requests multiple chapters, verse ranges, or disjoint selections (including
conjunctions like "and" or comma/semicolon lists), preserve them exactly as written. If the system has constraints
(for example, only a single chapter can be processed at a time), do NOT modify the user's message to fit those
constraints — leave the message intact and let downstream nodes handle any rejection or guidance. For translation
requests that specify a target language, leave the request as-is even if the language is unsupported. Only block or
rephrase explicit abuse.

# Output schema
Return structured output with keys: new_message (str), reason_for_decision (str), message_changed (bool).

# Examples

## Example 1

<past_conversation>
    user_message: Summarize John 1.
    assistant_response: Here's a summary of John 1.
</past_conversation>

<current_message>
    user_message: What is the context for John 1?
</current_message>

<assistant_response>
    new_message: What is the context for John 1?
    reason_for_decision: The intent was clear with no errors to fix.
    message_changed: False
</assistant_response>

## Example 2

<past_conversation>
    user_message: Show me "Jon 3:16"
    assistant_response: Displayed John 3:16.
</past_conversation>

<current_message>
    user_message: show me Jonh 3:17-18
</current_message>

<assistant_response>
    new_message: show me John 3:17-18
    reason_for_decision: Corrected spelling of "John".
    message_changed: True
</assistant_response>

## Example 3

<past_conversation>
    user_message: Explain John 1:1
    assistant_response: John claims that Jesus, the Word, existed in the beginning with God the Father.
</past_conversation>

<current_message>
    user_message: Explain John 1:3
</current_message>
    
<assistant_response>
    new_message: Explain John 1:3.
    reason_for_decision: The word 'John' was misspelled in the message.
    message_changed: True
</assistant_response>
"""


class PreprocessorResult(BaseModel):
    """Result type for the preprocessor node output."""

    new_message: str
    reason_for_decision: str
    message_changed: bool


def preprocess_user_query(state: Any) -> dict:  # pylint: disable=too-many-locals
    """Lightly clarify or correct the user's query using conversation history."""

    s = cast(BrainState, state)
    query = s["user_query"]
    chat_history = s["user_chat_history"]
    history_context_message = f"past_conversation: {json.dumps(chat_history)}"
    messages: list[EasyInputMessageParam] = [
        {"role": "user", "content": history_context_message},
        {"role": "user", "content": f"current_message: {query}"},
    ]
    response = open_ai_client.responses.parse(
        model="gpt-4o",
        instructions=PREPROCESSOR_AGENT_SYSTEM_PROMPT,
        input=cast(Any, messages),
        text_format=PreprocessorResult,
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
    preprocessor_result = cast(PreprocessorResult | None, response.output_parsed)
    if preprocessor_result is None:
        new_message = query
        reason_for_decision = "no changes"
        message_changed = False
    else:
        new_message = preprocessor_result.new_message
        reason_for_decision = preprocessor_result.reason_for_decision
        message_changed = preprocessor_result.message_changed
    logger.info(
        "new_message: %s\nreason_for_decision: %s\nmessage_changed: %s",
        new_message,
        reason_for_decision,
        message_changed,
    )
    return {"transformed_query": new_message if message_changed else query}


__all__ = [
    "PreprocessorResult",
    "PREPROCESSOR_AGENT_SYSTEM_PROMPT",
    "preprocess_user_query",
]
