"""Utilities to load the Berean Standard Bible into ChromaDB (hardened)."""

import json
import re
import time
import uuid
from typing import Dict, Iterable, List, Any, cast

import requests
import chromadb
from chromadb.api import ClientAPI as ChromaClientAPI  # public client interface type
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from openai import OpenAI
from openai import APIError, RateLimitError

# Typing helper for response_format without importing private names
ResponseFormatType = Any

from logger import get_logger

logger = get_logger(__name__)

VERSE_RE = re.compile(
    r"^(?P<book>[0-9A-Za-z ]+) (?P<chapter>\d+):(?P<verse>\d+) (?P<text>.+)$"
)

# ---------- OpenAI client (singleton) ----------
_client = OpenAI()  # uses env var OPENAI_API_KEY

# Default models
CHUNK_MODEL = "gpt-5-mini"
EMBED_MODEL = "text-embedding-3-large"

# Chunking controls
MAX_OUTPUT_TOKENS = 4000  # generous headroom
# Simple char-based guardrail to avoid giant prompts (tune as needed)
MAX_INPUT_CHARS = 18000


def fetch_verses() -> List[Dict[str, str]]:
    """Fetch the BSB text and parse into verse records."""
    url = "https://bereanbible.com/bsb.txt"
    logger.info("Fetching %s", url)
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()

    verses: List[Dict[str, str]] = []
    for line in resp.text.splitlines():
        match = VERSE_RE.match(line)
        if not match:
            continue
        book = match.group("book")
        chapter = match.group("chapter")
        verse = match.group("verse")
        text = match.group("text").strip()
        ref = f"{book} {chapter}:{verse}"
        verses.append(
            {
                "id": str(uuid.uuid4()),
                "book": book,
                "chapter": chapter,
                "verse": verse,
                "ref": ref,
                "text": text,
            }
        )
    logger.info("Parsed %d verses", len(verses))
    return verses


_SYSTEM = (
    "You group contiguous Bible verses into coherent sections.\n"
    "- Use ONLY the provided verses; do not invent text or references.\n"
    "- Chunks must be contiguous and in order; no overlaps.\n"
    "- For each chunk, set 'ref' to a compact range like 'John 1:1–5'.\n"
    "- Return ONLY JSON (no markdown)."
)

_USER_INSTRUCTIONS = (
    "Group the following Bible verses into semantic chunks.\n"
    "Requirements:\n"
    "1) Chunks are contiguous runs of the input verses.\n"
    "2) Do not overlap or reorder.\n"
    "3) Prefer chunk lengths of 2–6 verses when possible (but adapt as needed).\n"
    "4) Each chunk 'text' is the concatenation of its verses.\n\n"
    "Output JSON schema:\n"
    "{\n"
    '  "chunks": [\n'
    '    {"ref": "Book chap:verse–verse", "text": "concatenated text"}\n'
    "  ]\n"
    "}\n"
    "Return ONLY the JSON object."
)


def _safe_chat_json(messages, model: str, max_retries: int = 2) -> Dict:
    """Call the chat API with strict JSON response format + retries."""
    for attempt in range(max_retries + 1):
        try:
            rf = cast(ResponseFormatType, {"type": "json_object"})
            resp = _client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0,
                response_format=rf,
                max_tokens=MAX_OUTPUT_TOKENS,
            )
            content = (resp.choices[0].message.content or "").strip()
            return json.loads(content)
        except (json.JSONDecodeError, AttributeError) as e:
            logger.warning("JSON parse failed: %s", e)
        except (RateLimitError, APIError) as e:
            logger.warning("OpenAI API error (attempt %d): %s", attempt + 1, e)
            if attempt < max_retries:
                time.sleep(1.5 * (attempt + 1))
                continue
            raise
        # Backoff for JSON hiccups as well
        if attempt < max_retries:
            time.sleep(0.8 * (attempt + 1))
    return {"chunks": []}


def _batch_join_lines(verse_lines: List[str], max_chars: int) -> List[str]:
    """Split verse lines into batches that stay under max_chars."""
    batches: List[str] = []
    cur: List[str] = []
    cur_len = 0
    for line in verse_lines:
        add_len = len(line) + 1  # for newline
        if cur and (cur_len + add_len > max_chars):
            batches.append("\n".join(cur))
            cur = [line]
            cur_len = add_len
        else:
            cur.append(line)
            cur_len += add_len
    if cur:
        batches.append("\n".join(cur))
    return batches


def _normalize_chunks(chunks: List[Any]) -> List[Dict[str, str]]:
    """Ensure each chunk has id/ref/text as non-empty strings."""
    out: List[Dict[str, str]] = []
    for ch in chunks or []:
        if not isinstance(ch, dict):
            continue
        ref = ch.get("ref")
        text = ch.get("text")
        if isinstance(ref, str) and isinstance(text, str):
            ref = ref.strip()
            text = text.strip()
            if ref and text:
                out.append({"id": str(uuid.uuid4()), "ref": ref, "text": text})
    return out


def group_semantic_chunks(
    verses: Iterable[Dict[str, str]],
    model: str = CHUNK_MODEL,
) -> List[Dict[str, str]]:
    """
    Group contiguous verses into semantic chunks using an OpenAI model.

    Input item: {"ref": "John 1:1", "text": "..."}
    Returns: [{"id": "...", "ref": "John 1:1–5", "text": "..."}, ...]
    """
    verse_list = list(verses)
    if not verse_list:
        return []

    verse_lines = [f"{v['ref']} {v['text']}" for v in verse_list]

    # If the whole passage is modest, one shot; else batch to be safe.
    batches = _batch_join_lines(verse_lines, max_chars=MAX_INPUT_CHARS)

    all_chunks: List[Dict[str, str]] = []
    for i, batch in enumerate(batches):
        messages = [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": _USER_INSTRUCTIONS + "\n\n" + batch},
        ]
        data = _safe_chat_json(messages, model=model)
        chunks = _normalize_chunks(data.get("chunks", []))
        logger.info("Batch %d/%d produced %d chunks", i + 1, len(batches), len(chunks))
        all_chunks.extend(chunks)

    logger.info("Generated %d semantic chunks total", len(all_chunks))
    return all_chunks


def _recreate_collection(client: ChromaClientAPI, name: str):
    """Delete if exists, then create fresh collection with explicit embedding model."""
    # Avoid broad exceptions: check capability before calling
    existing: List[str] = []
    list_fn = getattr(client, "list_collections", None)
    if callable(list_fn):
        collections = cast(Iterable[Any], list_fn())
        existing = [
            name
            for c in collections
            if isinstance((name := getattr(c, "name", None)), str)
        ]
    if name in existing:
        client.delete_collection(name)
    return client.create_collection(
        name=name,
        embedding_function=OpenAIEmbeddingFunction(model_name=EMBED_MODEL),
    )


def load():
    """Load verses and semantic chunks into ChromaDB."""
    verses = fetch_verses()
    semantic_chunks = group_semantic_chunks(verses, model=CHUNK_MODEL)

    client = chromadb.PersistentClient(path="./data")
    collection = _recreate_collection(client, "bibles")

    # Insert verses
    collection.add(
        ids=[v["id"] for v in verses],
        documents=[v["text"] for v in verses],
        metadatas=[
            {
                "ref": v["ref"],
                "book": v["book"],
                "chapter": v["chapter"],
                "verse": v["verse"],
                "type": "verse",
            }
            for v in verses
        ],
    )

    # Insert semantic chunks
    collection.add(
        ids=[c["id"] for c in semantic_chunks],
        documents=[c["text"] for c in semantic_chunks],
        metadatas=[{"ref": c["ref"], "type": "semantic"} for c in semantic_chunks],
    )

    logger.info(
        "Inserted %d verses and %d semantic chunks",
        len(verses),
        len(semantic_chunks),
    )


def main():
    load()


if __name__ == "__main__":
    main()

