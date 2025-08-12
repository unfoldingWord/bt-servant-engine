import json
from types import SimpleNamespace
from pathlib import Path
import sys
import os
import chromadb

sys.path.append(str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("OPENAI_API_KEY", "test")
os.environ.setdefault("GROQ_API_KEY", "test")
os.environ.setdefault("META_VERIFY_TOKEN", "test")
os.environ.setdefault("META_WHATSAPP_TOKEN", "test")
os.environ.setdefault("META_PHONE_NUMBER_ID", "test")
os.environ.setdefault("META_APP_SECRET", "test")
os.environ.setdefault("FACEBOOK_USER_AGENT", "test")
os.environ.setdefault("BASE_URL", "http://example.com")
from db_loaders import load_bsb


class DummyResponse:
    def __init__(self, text: str):
        self.text = text

    def raise_for_status(self):
        pass


def test_genesis_processing(monkeypatch, tmp_path):
    sample_text = (
        "Genesis 1:1 In the beginning God created the heavens and the earth.\n"
        "Genesis 1:2 Now the earth was formless and void, and darkness was over the surface of the deep.\n"
        "Exodus 1:1 These are the names of the sons of Israel who went to Egypt with Jacob, each with his family.\n"
    )

    def fake_get(url, timeout):
        return DummyResponse(sample_text)

    monkeypatch.setattr(load_bsb.requests, "get", fake_get)

    class FakeClient:
        def __init__(self):
            self.responses = SimpleNamespace(
                create=lambda **kwargs: SimpleNamespace(
                    output=[
                        SimpleNamespace(
                            content=[
                                SimpleNamespace(
                                    text=json.dumps(
                                        {
                                            "chunks": [
                                                {
                                                    "ref": "Genesis 1:1-2",
                                                    "text": "In the beginning...",
                                                }
                                            ]
                                        }
                                    )
                                )
                            ]
                        )
                    ]
                )
            )

    monkeypatch.setattr(load_bsb, "OpenAI", FakeClient)

    verses = load_bsb.fetch_verses()
    genesis_verses = []
    for verse in verses:
        if verse["book"] != "Genesis":
            break
        genesis_verses.append(verse)

    assert len(genesis_verses) == 2
    assert genesis_verses[0]["ref"] == "Genesis 1:1"
    assert genesis_verses[-1]["ref"] == "Genesis 1:2"

    semantic_chunks = load_bsb.group_semantic_chunks(genesis_verses)
    assert semantic_chunks[0]["ref"] == "Genesis 1:1-2"

    class DummyEmbeddingFunction:
        def __call__(self, input):
            return [[0.0, 0.0, 0.0] for _ in input]

    client = chromadb.PersistentClient(path=str(tmp_path))
    collection = client.create_collection(
        "bibles", embedding_function=DummyEmbeddingFunction()
    )
    collection.add(
        ids=[v["id"] for v in genesis_verses],
        documents=[v["text"] for v in genesis_verses],
        metadatas=[
            {
                "ref": v["ref"],
                "book": v["book"],
                "chapter": v["chapter"],
                "verse": v["verse"],
                "type": "verse",
            }
            for v in genesis_verses
        ],
    )
    collection.add(
        ids=[c["id"] for c in semantic_chunks],
        documents=[c["text"] for c in semantic_chunks],
        metadatas=[{"ref": c["ref"], "type": "semantic"} for c in semantic_chunks],
    )

    verses_in_db = collection.get(where={"type": "verse"})
    assert len(verses_in_db["ids"]) == len(genesis_verses)
    assert all(md["book"] == "Genesis" for md in verses_in_db["metadatas"])
