"""Utilities to load the Berean Standard Bible into ChromaDB."""

import json
import re
import uuid
from typing import Dict, Iterable, List

import requests
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from openai import OpenAI

from logger import get_logger

logger = get_logger(__name__)

VERSE_RE = re.compile(r"^(?P<book>[0-9A-Za-z ]+) (?P<chapter>\d+):(?P<verse>\d+) (?P<text>.+)$")


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


def group_semantic_chunks(verses: Iterable[Dict[str, str]]) -> List[Dict[str, str]]:
    """Use GPT-5 to group contiguous verses into semantic chunks."""
    client = OpenAI()
    verse_lines = [f"{v['ref']} {v['text']}" for v in verses]
    prompt = (
        "Group the following Bible verses into semantic chunks. "
        "Return a JSON array of objects with 'ref' and 'text'."
    )
    response = client.responses.create(
        model="gpt-5-mini",
        reasoning={"effort": "low"},
        input=[
            {"role": "system", "content": "You group verses into coherent sections."},
            {"role": "user", "content": "\n".join(verse_lines)},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
    )
    content = response.output[0].content[0].text  # type: ignore[attr-defined]
    data = json.loads(content)
    chunks = data.get("chunks", [])
    results: List[Dict[str, str]] = []
    for chunk in chunks:
        ref = chunk.get("ref")
        text = chunk.get("text")
        if not (ref and text):
            continue
        results.append({"id": str(uuid.uuid4()), "ref": ref, "text": text})
    logger.info("Generated %d semantic chunks", len(results))
    return results


def load():
    """Load verses and semantic chunks into ChromaDB."""
    verses = fetch_verses()
    semantic_chunks = group_semantic_chunks(verses)

    client = chromadb.PersistentClient(path="./data")
    try:
        client.delete_collection("bibles")
    except Exception:
        logger.warning("No existing collection to delete")

    collection = client.create_collection(
        name="bibles",
        embedding_function=OpenAIEmbeddingFunction(),
    )

    collection.add(
        ids=[v["id"] for v in verses],
        documents=[v["text"] for v in verses],
        metadatas=[{"ref": v["ref"], "book": v["book"], "chapter": v["chapter"], "verse": v["verse"], "type": "verse"} for v in verses],
    )

    collection.add(
        ids=[c["id"] for c in semantic_chunks],
        documents=[c["text"] for c in semantic_chunks],
        metadatas=[{"ref": c["ref"], "type": "semantic"} for c in semantic_chunks],
    )

    logger.info("Inserted %d verses and %d semantic chunks", len(verses), len(semantic_chunks))


def main():
    load()


if __name__ == "__main__":
    main()
