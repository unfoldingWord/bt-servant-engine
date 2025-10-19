"""Helper utilities for administrative Chroma merge operations."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


def iter_collection_batches(
    collection: Any,
    *,
    batch_size: int = 1000,
    include_embeddings: bool = False,
) -> list[dict[str, Any]]:
    """Return a list of batch dictionaries from a collection handle."""
    if collection is None:
        return []
    kwargs: dict[str, Any] = {"limit": batch_size}
    includes: list[str] = ["documents", "metadatas"]
    if include_embeddings:
        includes.append("embeddings")
    kwargs_with_include = dict(kwargs)
    kwargs_with_include["include"] = includes

    offset = 0
    batches: list[dict[str, Any]] = []
    while True:
        try:
            result = collection.get(offset=offset, **kwargs_with_include)
        except (TypeError, ValueError):
            try:
                result = collection.get(**kwargs)
            except (TypeError, ValueError):
                result = collection.get()
        ids = result.get("ids") or []
        if not ids:
            break
        batches.append(result)
        if len(ids) < batch_size:
            break
        offset += batch_size
    return batches


def estimate_tokens(text: str | None) -> int:
    """Rough token estimate from text length."""
    if not text:
        return 1
    return max(1, len(text) // 4)


def yield_token_limited_slices(
    ids: list[str],
    docs: list[str] | None,
    metas: list[dict[str, Any]] | None,
    *,
    max_tokens: int,
) -> list[tuple[list[str], list[str], list[dict[str, Any]] | None]]:
    """Split aligned (ids, docs, metas) into sub-batches under a token budget."""
    if not ids:
        return []
    if docs is None:
        return [(ids, [], metas)]

    out: list[tuple[list[str], list[str], list[dict[str, Any]] | None]] = []
    cur_ids: list[str] = []
    cur_docs: list[str] = []
    cur_metas: list[dict[str, Any]] | None = [] if metas is not None else None
    budget = 0

    for i, did in enumerate(ids):
        doc = docs[i]
        tokens = estimate_tokens(doc)
        if cur_ids and budget + tokens > max_tokens:
            out.append((cur_ids, cur_docs, cur_metas))
            cur_ids = []
            cur_docs = []
            cur_metas = [] if metas is not None else None
            budget = 0
        cur_ids.append(did)
        cur_docs.append(doc)
        if metas is not None:
            cur_metas.append(metas[i])  # type: ignore[union-attr]
        budget += tokens

    if cur_ids:
        out.append((cur_ids, cur_docs, cur_metas))
    return out


def now_iso() -> str:
    """Return the current UTC timestamp in ISO format."""
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class MetadataTaggingConfig:
    """Configuration inputs for stamping merge metadata."""

    enabled: bool
    tag_key: str
    source: str
    task_id: str
    tag_timestamp: bool
    source_ids: list[str] | None = None
    source_id_key: str = "_source_id"


def apply_metadata_tags(
    metadatas: list[dict[str, Any]] | None, config: MetadataTaggingConfig
) -> list[dict[str, Any]] | None:
    """Stamp merge metadata onto documents when enabled."""
    if not config.enabled:
        return metadatas
    if metadatas is None:
        return None
    stamped: list[dict[str, Any]] = []
    timestamp = now_iso() if config.tag_timestamp else None
    for idx, metadata in enumerate(metadatas):
        item = dict(metadata) if metadata is not None else {}
        item[config.tag_key] = config.source
        item["_merge_task_id"] = config.task_id
        if timestamp is not None:
            item["_merged_at"] = timestamp
        if config.source_ids is not None and 0 <= idx < len(config.source_ids):
            item[config.source_id_key] = config.source_ids[idx]
        stamped.append(item)
    return stamped


def compute_duplicate_preview(
    source_col: Any,
    dest_col: Any,
    *,
    limit: int,
    batch_size: int,
) -> tuple[bool, list[str]]:
    """Return (duplicates_found, preview_ids_up_to_limit)."""
    preview: list[str] = []
    dest_ids: set[str] = set()
    for batch in iter_collection_batches(dest_col, batch_size=10000, include_embeddings=False):
        for _id in batch.get("ids") or []:
            dest_ids.add(str(_id))
    if not dest_ids:
        return False, []
    for batch in iter_collection_batches(
        source_col, batch_size=batch_size, include_embeddings=False
    ):
        for _id in batch.get("ids") or []:
            sid = str(_id)
            if sid in dest_ids:
                preview.append(sid)
                if len(preview) >= limit:
                    return True, preview
    return (len(preview) > 0), preview


__all__ = [
    "MetadataTaggingConfig",
    "apply_metadata_tags",
    "compute_duplicate_preview",
    "estimate_tokens",
    "iter_collection_batches",
    "now_iso",
    "yield_token_limited_slices",
]
