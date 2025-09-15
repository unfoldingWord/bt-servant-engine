from __future__ import annotations

import json
from typing import Any, Callable, List, cast

from openai import OpenAI, OpenAIError
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam

from config import config
from logger import get_logger


logger = get_logger(__name__)


def chunk_message(  # noqa: C901
    state: Any,
    *,
    chop_text: Callable[..., list[str]],
    combine_chunks: Callable[..., list[str]],
    CHOP_AGENT_SYSTEM_PROMPT: str,
    open_ai_client: OpenAI,
    add_tokens: Callable[..., None],
    extract_cached_input_tokens: Callable[[Any], int | None],
) -> dict:
    """Chunk oversized responses to respect WhatsApp limits, via LLM or fallback."""
    logger.info("MESSAGE TOO BIG. CHUNKING...")
    s = cast(dict[str, Any], state)
    responses = cast(list[str], s["translated_responses"])
    text_to_chunk = responses[0]
    chunk_max = config.MAX_META_TEXT_LENGTH - 100
    try:
        chat_messages = cast(List[ChatCompletionMessageParam], [
            {
                "role": "system",
                "content": CHOP_AGENT_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": f"text to chop: \n\n{text_to_chunk}",
            },
        ])
        completion = open_ai_client.chat.completions.create(
            model='gpt-4o',
            messages=chat_messages,
        )
        usage = getattr(completion, "usage", None)
        if usage is not None:
            it = getattr(usage, "prompt_tokens", None)
            ot = getattr(usage, "completion_tokens", None)
            tt = getattr(usage, "total_tokens", None)
            cit = extract_cached_input_tokens(usage)
            add_tokens(it, ot, tt, model="gpt-4o", cached_input_tokens=cit)
        response_content = completion.choices[0].message.content
        if not isinstance(response_content, str):
            raise ValueError("empty or non-text content from chat completion")
        chunks = json.loads(response_content)
    except (OpenAIError, json.JSONDecodeError, ValueError):
        logger.error("LLM chunking failed. Falling back to deterministic chunking.", exc_info=True)
        chunks = None

    def _pack_items(items: list[str], max_len: int) -> list[str]:
        out: list[str] = []
        cur = ""
        for it in items:
            sep = (", " if cur else "")
            if len(cur) + len(sep) + len(it) <= max_len:
                cur += sep + it
            else:
                if cur:
                    out.append(cur)
                if len(it) <= max_len:
                    cur = it
                else:
                    for j in range(0, len(it), max_len):
                        out.append(it[j:j+max_len])
                    cur = ""
        if cur:
            out.append(cur)
        return out

    if not isinstance(chunks, list) or any(not isinstance(c, str) for c in chunks):
        if text_to_chunk.count(",") >= 10:
            parts = [p.strip() for p in text_to_chunk.split(",") if p.strip()]
            chunks = _pack_items(parts, chunk_max)
        else:
            chunks = chop_text(text=text_to_chunk, n=chunk_max)
    else:
        fixed: list[str] = []
        for c in chunks:
            if len(c) <= chunk_max:
                fixed.append(c)
            else:
                if c.count(",") >= 10:
                    parts = [p.strip() for p in c.split(",") if p.strip()]
                    fixed.extend(_pack_items(parts, chunk_max))
                else:
                    fixed.extend(chop_text(text=c, n=chunk_max))
        chunks = fixed

    chunks.extend(responses[1:])
    return {"translated_responses": combine_chunks(chunks=chunks, chunk_max=chunk_max)}
