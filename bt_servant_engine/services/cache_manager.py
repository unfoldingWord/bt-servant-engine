"""Caching infrastructure for intent handlers with disk and memory backends."""

# pylint: disable=missing-function-docstring,too-many-instance-attributes,too-many-locals,too-many-branches,redefined-outer-name

from __future__ import annotations

import hashlib
import json
import shutil
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable

from bt_servant_engine.core.config import config
from bt_servant_engine.core.intents import IntentType
from bt_servant_engine.core.logging import get_logger
from utils.perf import get_current_trace, record_external_span

logger = get_logger(__name__)

CACHE_SCHEMA_VERSION = "2025-03-branch-v1"
_INTENT_SENTINEL = "__cache_intent__"
_TUPLE_SENTINEL = "__cache_tuple__"


def _hash_key(key: Any) -> tuple[str, str]:
    """Return a stable hash for arbitrary key objects."""
    key_repr = repr(key)
    digest = hashlib.sha256(key_repr.encode("utf-8")).hexdigest()
    return digest, key_repr


def _encode_payload(value: Any) -> bytes:
    def _default(obj: Any) -> Any:
        if isinstance(obj, IntentType):
            return {_INTENT_SENTINEL: obj.value}
        if isinstance(obj, tuple):
            return {_TUPLE_SENTINEL: list(obj)}
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    text = json.dumps(value, default=_default, ensure_ascii=False, separators=(",", ":"))
    return text.encode("utf-8")


def _decode_payload(data: bytes) -> Any:
    def _object_hook(obj: dict[str, Any]) -> Any:
        if len(obj) == 1 and _INTENT_SENTINEL in obj:
            return IntentType(obj[_INTENT_SENTINEL])
        if len(obj) == 1 and _TUPLE_SENTINEL in obj:
            return tuple(obj[_TUPLE_SENTINEL])
        return obj

    text = data.decode("utf-8")
    return json.loads(text, object_hook=_object_hook)


def _perf_span(name: str, start: float, end: float) -> None:
    trace = get_current_trace()
    if trace:
        record_external_span(name, start, end, trace_id=trace)


@dataclass(slots=True)
class CacheStats:
    """Mutable counters for cache operations."""

    hits: int = 0
    misses: int = 0
    stores: int = 0
    evictions: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "stores": self.stores,
            "evictions": self.evictions,
        }


@dataclass(slots=True)
class CacheEntryMeta:
    """Metadata describing a stored cache entry."""

    key: str
    key_repr: str
    created_at: float
    expires_at: float
    last_access: float
    size_bytes: int

    def is_expired(self, now: float) -> bool:
        if self.expires_at < 0:
            return False
        return now >= self.expires_at


class MemoryCacheBackend:
    """In-memory cache backend with LRU eviction."""

    def __init__(self, max_entries: int | None) -> None:
        self._lock = threading.RLock()
        self._entries: "OrderedDict[str, tuple[Any, CacheEntryMeta]]" = OrderedDict()
        self._max_entries = max_entries

    def get(self, key: str, now: float) -> tuple[Any, CacheEntryMeta] | None:
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            value, meta = entry
            if meta.is_expired(now):
                self._entries.pop(key, None)
                return None
            meta.last_access = now
            self._entries.move_to_end(key)
            return value, meta

    def set(self, key: str, value: Any, meta: CacheEntryMeta) -> None:
        with self._lock:
            self._entries[key] = (value, meta)
            self._entries.move_to_end(key)
            self._evict_if_needed()

    def delete(self, key: str) -> None:
        with self._lock:
            self._entries.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def entries(self) -> Iterable[CacheEntryMeta]:
        with self._lock:
            return [meta for _, meta in self._entries.values()]

    def prune_older_than(self, cutoff: float) -> list[CacheEntryMeta]:
        removed: list[CacheEntryMeta] = []
        with self._lock:
            keys = list(self._entries.keys())
            for key in keys:
                _, meta = self._entries[key]
                if meta.created_at < cutoff:
                    removed.append(meta)
                    self._entries.pop(key, None)
        return removed

    def _evict_if_needed(self) -> list[CacheEntryMeta]:
        evicted: list[CacheEntryMeta] = []
        if self._max_entries is None:
            return evicted
        while len(self._entries) > self._max_entries:
            _, (_, meta) = self._entries.popitem(last=False)
            evicted.append(meta)
        return evicted


class DiskCacheBackend:
    """Disk-backed cache storing serialized values with manifest bookkeeping."""

    def __init__(self, cache_dir: Path, max_bytes: int) -> None:
        self._cache_dir = cache_dir
        self._manifest_path = cache_dir / "manifest.json"
        self._lock = threading.RLock()
        self._max_bytes = max_bytes
        self._entries: Dict[str, CacheEntryMeta] = {}
        self._total_bytes = 0
        self._load_manifest()

    def _load_manifest(self) -> None:
        if not self._manifest_path.exists():
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            self._persist_manifest()
            return
        try:
            raw = json.loads(self._manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            logger.warning("[cache] manifest load failed for %s; rebuilding", self._manifest_path)
            shutil.rmtree(self._cache_dir, ignore_errors=True)
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            self._persist_manifest()
            return
        entries = raw.get("entries", {})
        total_bytes = 0
        for key, meta in entries.items():
            path = self._cache_dir / f"{key}.json"
            if not path.exists():
                continue
            cem = CacheEntryMeta(
                key=key,
                key_repr=meta.get("key_repr", key),
                created_at=float(meta.get("created_at", time.time())),
                expires_at=float(meta.get("expires_at", 0)),
                last_access=float(meta.get("last_access", 0)),
                size_bytes=int(meta.get("size_bytes", path.stat().st_size)),
            )
            self._entries[key] = cem
            total_bytes += cem.size_bytes
        self._total_bytes = total_bytes

    def _persist_manifest(self) -> None:
        data = {
            "entries": {
                key: {
                    "key_repr": meta.key_repr,
                    "created_at": meta.created_at,
                    "expires_at": meta.expires_at,
                    "last_access": meta.last_access,
                    "size_bytes": meta.size_bytes,
                }
                for key, meta in self._entries.items()
            },
            "total_bytes": self._total_bytes,
        }
        tmp = self._manifest_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        tmp.replace(self._manifest_path)

    def get(self, key: str, now: float) -> tuple[Any, CacheEntryMeta] | None:
        with self._lock:
            meta = self._entries.get(key)
            if meta is None:
                return None
            if meta.is_expired(now):
                self._delete_unlocked(key)
                return None
            path = self._cache_dir / f"{key}.json"
            try:
                with path.open("rb") as fh:
                    payload = fh.read()
                value = _decode_payload(payload)
            except (OSError, json.JSONDecodeError, ValueError, TypeError):
                logger.warning("[cache] failed to load cache entry %s; removing", key)
                self._delete_unlocked(key)
                return None
            meta.last_access = now
            self._persist_manifest()
            return value, meta

    def set(self, key: str, value: Any, meta: CacheEntryMeta) -> list[CacheEntryMeta]:
        payload = _encode_payload(value)
        meta.size_bytes = len(payload)
        with self._lock:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            path = self._cache_dir / f"{key}.json"
            tmp = path.with_suffix(".tmp")
            with tmp.open("wb") as fh:
                fh.write(payload)
            tmp.replace(path)
            self._entries[key] = meta
            self._total_bytes = sum(entry.size_bytes for entry in self._entries.values())
            evicted = self._evict_if_needed_unlocked()
            self._persist_manifest()
            return evicted

    def delete(self, key: str) -> None:
        with self._lock:
            self._delete_unlocked(key)
            self._persist_manifest()

    def _delete_unlocked(self, key: str) -> None:
        meta = self._entries.pop(key, None)
        if meta is None:
            return
        path = self._cache_dir / f"{key}.json"
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        self._total_bytes = max(0, self._total_bytes - meta.size_bytes)

    def clear(self) -> None:
        with self._lock:
            shutil.rmtree(self._cache_dir, ignore_errors=True)
            self._entries.clear()
            self._total_bytes = 0
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            self._persist_manifest()

    def entries(self) -> Iterable[CacheEntryMeta]:
        with self._lock:
            return list(self._entries.values())

    def total_bytes(self) -> int:
        with self._lock:
            return self._total_bytes

    def prune_older_than(self, cutoff: float) -> list[CacheEntryMeta]:
        removed: list[CacheEntryMeta] = []
        with self._lock:
            keys = list(self._entries.keys())
            for key in keys:
                meta = self._entries.get(key)
                if meta and meta.created_at < cutoff:
                    self._delete_unlocked(key)
                    removed.append(meta)
            self._persist_manifest()
        return removed

    def _evict_if_needed_unlocked(self) -> list[CacheEntryMeta]:
        evicted: list[CacheEntryMeta] = []
        while self._total_bytes > self._max_bytes and self._entries:
            key, meta = min(self._entries.items(), key=lambda item: item[1].last_access)
            self._delete_unlocked(key)
            evicted.append(meta)
        return evicted


@dataclass(slots=True)
class CacheConfig:
    """Configuration for an individual cache namespace."""

    name: str
    ttl_seconds: int
    max_entries: int | None


class CacheStore:
    """High-level cache interface with logging and perf instrumentation."""

    def __init__(
        self,
        config: CacheConfig,
        backend: MemoryCacheBackend | DiskCacheBackend,
        *,
        enabled: bool,
    ) -> None:
        self._config = config
        self._backend = backend
        self._enabled = enabled
        self._stats = CacheStats()
        self._lock = threading.RLock()

    @property
    def name(self) -> str:
        return self._config.name

    def get_or_set(
        self,
        key: Any,
        compute: Callable[[], Any],
        *,
        should_store: Callable[[Any], bool] | None = None,
    ) -> tuple[Any, bool]:
        if not self._enabled:
            value = compute()
            return value, False
        key_hash, key_repr = _hash_key(key)
        now = time.time()
        start = time.time()
        cached = self._backend.get(key_hash, now)
        duration = time.time() - start
        if cached:
            self._stats.hits += 1
            _, meta = cached
            logger.info(
                "[cache] hit name=%s key=%s size=%s age=%.1fs",
                self.name,
                key_repr,
                meta.size_bytes,
                now - meta.created_at,
            )
            _perf_span(f"cache_hit:{self.name}", start, start + duration)
            return cached[0], True
        self._stats.misses += 1
        logger.info("[cache] miss name=%s key=%s", self.name, key_repr)
        _perf_span(f"cache_miss:{self.name}", start, start + duration)
        value = compute()
        if should_store is not None and not should_store(value):
            logger.info(
                "[cache] store skipped name=%s key=%s (predicate)",
                self.name,
                key_repr,
            )
            return value, False
        now = time.time()
        expires_at = -1 if self._config.ttl_seconds < 0 else now + self._config.ttl_seconds
        meta = CacheEntryMeta(
            key=key_hash,
            key_repr=key_repr,
            created_at=now,
            expires_at=expires_at,
            last_access=now,
            size_bytes=0,
        )
        start_store = time.time()
        evicted: list[CacheEntryMeta] = []
        if isinstance(self._backend, MemoryCacheBackend):
            self._backend.set(key_hash, value, meta)
        else:
            evicted = self._backend.set(key_hash, value, meta)
        store_duration = time.time() - start_store
        self._stats.stores += 1
        self._stats.evictions += len(evicted)
        if evicted:
            logger.info(
                "[cache] eviction name=%s count=%d keys=%s",
                self.name,
                len(evicted),
                [m.key_repr for m in evicted],
            )
        logger.info(
            "[cache] store name=%s key=%s size=%s ttl=%ss",
            self.name,
            key_repr,
            meta.size_bytes,
            self._config.ttl_seconds,
        )
        if store_duration:
            _perf_span(f"cache_store:{self.name}", start_store, start_store + store_duration)
        return value, False

    def clear(self) -> None:
        logger.info("[cache] clear name=%s", self.name)
        self._backend.clear()
        self._stats = CacheStats()

    def stats(self) -> dict[str, Any]:
        entries = self._sorted_entries()
        newest = max((e.created_at for e in entries), default=None)
        oldest = min((e.created_at for e in entries), default=None)
        bytes_used = 0
        if isinstance(self._backend, DiskCacheBackend):
            bytes_used = self._backend.total_bytes()
        return {
            "name": self.name,
            "enabled": self._enabled,
            "ttl_seconds": self._config.ttl_seconds,
            "max_entries": self._config.max_entries,
            "entry_count": len(entries),
            "bytes_used": bytes_used,
            "oldest_entry_epoch": oldest,
            "newest_entry_epoch": newest,
            "stats": self._stats.as_dict(),
        }

    def detailed_stats(self, sample_limit: int = 10) -> dict[str, Any]:
        data = self.stats()
        entries = self._sorted_entries()
        samples = []
        now = time.time()
        for meta in entries[:sample_limit]:
            ttl_remaining = float("inf") if meta.expires_at < 0 else max(0.0, meta.expires_at - now)
            samples.append(
                {
                    "key_repr": meta.key_repr,
                    "size_bytes": meta.size_bytes,
                    "created_at": meta.created_at,
                    "expires_at": meta.expires_at,
                    "last_access": meta.last_access,
                    "age_seconds": max(0.0, now - meta.created_at),
                    "ttl_remaining": ttl_remaining,
                }
            )
        data["samples"] = samples
        return data

    def _sorted_entries(self) -> list[CacheEntryMeta]:
        entries = list(self._backend.entries())
        entries.sort(key=lambda e: e.last_access, reverse=True)
        return entries

    def prune_older_than(self, cutoff: float) -> int:
        if isinstance(self._backend, MemoryCacheBackend):
            removed = self._backend.prune_older_than(cutoff)
        elif isinstance(self._backend, DiskCacheBackend):
            removed = self._backend.prune_older_than(cutoff)
        else:
            removed = []
        count = len(removed)
        if count:
            self._stats.evictions += count
            logger.info(
                "[cache] prune name=%s removed=%d cutoff=%s",
                self.name,
                count,
                cutoff,
            )
        return count


class CacheManager:
    """Registry of caches shared across the app."""

    def __init__(
        self,
        *,
        enabled: bool,
        backend_type: str,
        disk_root: Path,
        disk_max_bytes: int,
    ) -> None:
        self._enabled = enabled
        self._backend_type = backend_type
        self._disk_root = disk_root
        self._disk_max_bytes = disk_max_bytes
        self._caches: Dict[str, CacheStore] = {}
        self._lock = threading.RLock()

    def register(self, cache_config: CacheConfig, *, cache_enabled: bool = True) -> CacheStore:
        with self._lock:
            if cache_config.name in self._caches:
                return self._caches[cache_config.name]
            backend = self._create_backend(cache_config)
            store = CacheStore(cache_config, backend, enabled=self._enabled and cache_enabled)
            self._caches[cache_config.name] = store
            return store

    def cache(self, name: str) -> CacheStore:
        with self._lock:
            return self._caches[name]

    def _create_backend(self, cache_config: CacheConfig) -> MemoryCacheBackend | DiskCacheBackend:
        if self._backend_type == "memory":
            return MemoryCacheBackend(cache_config.max_entries)
        cache_dir = self._disk_root / cache_config.name
        cache_dir.mkdir(parents=True, exist_ok=True)
        return DiskCacheBackend(cache_dir, max_bytes=self._disk_max_bytes)

    def clear_all(self) -> None:
        logger.info("[cache] clearing all caches")
        for cache in self._caches.values():
            cache.clear()
        if self._backend_type == "disk":
            shutil.rmtree(self._disk_root, ignore_errors=True)
            self._disk_root.mkdir(parents=True, exist_ok=True)

    def clear_cache(self, name: str) -> None:
        cache = self._caches.get(name)
        if not cache:
            raise KeyError(name)
        cache.clear()
        if self._backend_type == "disk":
            cache_dir = self._disk_root / name
            shutil.rmtree(cache_dir, ignore_errors=True)
            cache_dir.mkdir(parents=True, exist_ok=True)

    def prune_cache(self, name: str, cutoff: float) -> int:
        cache = self._caches.get(name)
        if not cache:
            raise KeyError(name)
        return cache.prune_older_than(cutoff)

    def prune_all(self, cutoff: float) -> dict[str, int]:
        result: dict[str, int] = {}
        for name, cache in self._caches.items():
            result[name] = cache.prune_older_than(cutoff)
        return result

    def stats(self) -> dict[str, Any]:
        caches = {name: cache.stats() for name, cache in self._caches.items()}
        return {
            "enabled": self._enabled,
            "backend": self._backend_type,
            "disk_root": str(self._disk_root),
            "disk_max_bytes": self._disk_max_bytes,
            "caches": caches,
        }

    def cache_names(self) -> list[str]:
        return list(self._caches.keys())


def _init_manager() -> CacheManager:
    disk_root = config.DATA_DIR / "cache"
    disk_root.mkdir(parents=True, exist_ok=True)
    manager = CacheManager(
        enabled=config.CACHE_ENABLED,
        backend_type=config.CACHE_BACKEND,
        disk_root=disk_root,
        disk_max_bytes=config.CACHE_DISK_MAX_BYTES,
    )
    manager.register(
        CacheConfig(
            name="selection",
            ttl_seconds=config.CACHE_SELECTION_TTL_SECONDS,
            max_entries=config.CACHE_SELECTION_MAX_ENTRIES,
        ),
        cache_enabled=config.CACHE_SELECTION_ENABLED,
    )
    manager.register(
        CacheConfig(
            name="passage_summary",
            ttl_seconds=config.CACHE_SUMMARY_TTL_SECONDS,
            max_entries=config.CACHE_SUMMARY_MAX_ENTRIES,
        ),
        cache_enabled=config.CACHE_SUMMARY_ENABLED,
    )
    manager.register(
        CacheConfig(
            name="passage_keywords",
            ttl_seconds=config.CACHE_KEYWORDS_TTL_SECONDS,
            max_entries=config.CACHE_KEYWORDS_MAX_ENTRIES,
        ),
        cache_enabled=config.CACHE_KEYWORDS_ENABLED,
    )
    manager.register(
        CacheConfig(
            name="translation_helps",
            ttl_seconds=config.CACHE_TRANSLATION_HELPS_TTL_SECONDS,
            max_entries=config.CACHE_TRANSLATION_HELPS_MAX_ENTRIES,
        ),
        cache_enabled=config.CACHE_TRANSLATION_HELPS_ENABLED,
    )
    manager.register(
        CacheConfig(
            name="rag_vector",
            ttl_seconds=config.CACHE_RAG_VECTOR_TTL_SECONDS,
            max_entries=config.CACHE_RAG_VECTOR_MAX_ENTRIES,
        ),
        cache_enabled=config.CACHE_RAG_VECTOR_ENABLED,
    )
    manager.register(
        CacheConfig(
            name="rag_final",
            ttl_seconds=config.CACHE_RAG_FINAL_TTL_SECONDS,
            max_entries=config.CACHE_RAG_FINAL_MAX_ENTRIES,
        ),
        cache_enabled=config.CACHE_RAG_FINAL_ENABLED,
    )
    return manager


cache_manager = _init_manager()


def get_cache(name: str) -> CacheStore:
    """Convenience accessor for registered caches."""
    return cache_manager.cache(name)


__all__ = [
    "CACHE_SCHEMA_VERSION",
    "CacheManager",
    "cache_manager",
    "get_cache",
]
