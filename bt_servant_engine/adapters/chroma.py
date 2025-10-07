"""Chroma adapter implementing the core storage port."""

from __future__ import annotations

from typing import Any, Iterable

from db import (
    CollectionExistsError,
    CollectionNotFoundError,
    DocumentNotFoundError,
    count_documents_in_collection,
    create_chroma_collection,
    delete_chroma_collection,
    delete_document,
    get_chroma_collection,
    get_chroma_collections_pair,
    get_document_text_and_metadata,
    iter_collection_batches,
    list_chroma_collections,
    list_document_ids_in_collection,
    max_numeric_id_in_collection,
)

from bt_servant_engine.core.ports import ChromaPort


class ChromaAdapter(ChromaPort):
    """Concrete adapter around the legacy Chroma helpers."""

    def list_collections(self) -> list[str]:
        return list_chroma_collections()

    def create_collection(self, name: str) -> None:
        create_chroma_collection(name)

    def delete_collection(self, name: str) -> None:
        delete_chroma_collection(name)

    def delete_document(self, name: str, document_id: str) -> None:
        delete_document(name, document_id)

    def count_documents(self, name: str) -> int:
        return count_documents_in_collection(name)

    def get_document_text_and_metadata(
        self, name: str, document_id: str
    ) -> tuple[str, dict[str, Any]]:
        text, metadata = get_document_text_and_metadata(name, document_id)
        return text, dict(metadata)

    def list_document_ids(self, name: str) -> list[str]:
        return list_document_ids_in_collection(name)

    def iter_batches(
        self,
        name: str,
        *,
        batch_size: int = 1000,
        include_embeddings: bool = False,
    ) -> Iterable[dict[str, Any]]:
        collection = get_chroma_collection(name)
        return iter_collection_batches(
            collection, batch_size=batch_size, include_embeddings=include_embeddings
        )

    def get_collections_pair(self, source: str, dest: str) -> tuple[Any, Any]:
        return get_chroma_collections_pair(source, dest)

    def max_numeric_id(self, name: str) -> int:
        return max_numeric_id_in_collection(name)


__all__ = [
    "ChromaAdapter",
    "CollectionExistsError",
    "CollectionNotFoundError",
    "DocumentNotFoundError",
]
