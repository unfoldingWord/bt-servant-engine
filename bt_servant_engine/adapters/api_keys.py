"""SQLite-backed adapter for API key storage."""

from __future__ import annotations

import secrets
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

import bcrypt

from bt_servant_engine.core.api_key_models import KEY_PREFIX_LENGTH, APIKey, Environment
from bt_servant_engine.core.config import config
from bt_servant_engine.core.ports import APIKeyPort

# Key format: bts_<env>_<random>
KEY_RANDOM_LENGTH = 20  # Random portion


class APIKeyAdapter(APIKeyPort):
    """SQLite-backed API key storage with in-memory validation cache."""

    def __init__(self, db_path: Path | None = None) -> None:
        """Initialize the adapter.

        Args:
            db_path: Path to the SQLite database file.
                    Defaults to DATA_DIR/api_keys.db.
        """
        data_dir = Path(getattr(config, "DATA_DIR", Path("/data")))
        data_dir.mkdir(parents=True, exist_ok=True)
        self._db_path = db_path or (data_dir / "api_keys.db")
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._init_db()
        # Cache: prefix -> (key_hash, key_id) for fast validation
        self._cache: dict[str, tuple[str, str]] = {}
        self._cache_lock = Lock()
        self._load_cache()

    def _get_conn(self) -> sqlite3.Connection:
        """Get a SQLite connection."""
        conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Initialize the database schema."""
        with self._lock, self._get_conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS api_keys (
                    id TEXT PRIMARY KEY,
                    key_prefix TEXT NOT NULL UNIQUE,
                    key_hash TEXT NOT NULL,
                    name TEXT NOT NULL,
                    environment TEXT NOT NULL DEFAULT 'prod',
                    created_at TEXT NOT NULL,
                    revoked_at TEXT,
                    last_used_at TEXT,
                    rate_limit_per_minute INTEGER DEFAULT 60,
                    expires_at TEXT,
                    CONSTRAINT env_check CHECK (environment IN ('prod', 'staging', 'dev'))
                );
                CREATE INDEX IF NOT EXISTS idx_api_keys_prefix ON api_keys(key_prefix);
            """)

    def _load_cache(self) -> None:
        """Load active keys into memory cache."""
        with self._lock, self._get_conn() as conn:
            rows = conn.execute(
                "SELECT key_prefix, key_hash, id FROM api_keys WHERE revoked_at IS NULL"
            ).fetchall()
        with self._cache_lock:
            self._cache = {
                row["key_prefix"]: (row["key_hash"], row["id"]) for row in rows
            }

    def _invalidate_cache(self, prefix: str | None = None) -> None:
        """Invalidate cache entry or reload all."""
        if prefix:
            with self._cache_lock:
                self._cache.pop(prefix, None)
        else:
            self._load_cache()

    @staticmethod
    def _generate_key(environment: str) -> tuple[str, str]:
        """Generate a new API key. Returns (full_key, prefix)."""
        random_part = secrets.token_urlsafe(KEY_RANDOM_LENGTH)[:KEY_RANDOM_LENGTH]
        full_key = f"bts_{environment}_{random_part}"
        prefix = full_key[:KEY_PREFIX_LENGTH]
        return full_key, prefix

    @staticmethod
    def _hash_key(raw_key: str) -> str:
        """Hash a key with bcrypt."""
        return bcrypt.hashpw(raw_key.encode(), bcrypt.gensalt()).decode()

    @staticmethod
    def _verify_key(raw_key: str, key_hash: str) -> bool:
        """Verify a key against its hash."""
        try:
            return bcrypt.checkpw(raw_key.encode(), key_hash.encode())
        except Exception:  # noqa: BLE001
            return False

    def create_key(
        self,
        name: str,
        environment: str,
        rate_limit_per_minute: int = 60,
        expires_at: datetime | None = None,
    ) -> tuple[APIKey, str]:
        """Create a new API key."""
        key_id = str(uuid.uuid4())
        raw_key, prefix = self._generate_key(environment)
        key_hash = self._hash_key(raw_key)
        now = datetime.now(timezone.utc).isoformat()

        with self._lock, self._get_conn() as conn:
            conn.execute(
                """INSERT INTO api_keys
                   (id, key_prefix, key_hash, name, environment, created_at,
                    rate_limit_per_minute, expires_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    key_id,
                    prefix,
                    key_hash,
                    name,
                    environment,
                    now,
                    rate_limit_per_minute,
                    expires_at.isoformat() if expires_at else None,
                ),
            )

        # Update cache
        with self._cache_lock:
            self._cache[prefix] = (key_hash, key_id)

        key = APIKey(
            id=key_id,
            key_prefix=prefix,
            name=name,
            environment=Environment(environment),
            created_at=datetime.fromisoformat(now),
            rate_limit_per_minute=rate_limit_per_minute,
            expires_at=expires_at,
        )
        return key, raw_key

    def validate_key(self, raw_key: str) -> APIKey | None:
        """Fast validation using cache."""
        if not raw_key or not raw_key.startswith("bts_"):
            return None

        prefix = raw_key[:KEY_PREFIX_LENGTH]

        # Check cache first
        with self._cache_lock:
            cached = self._cache.get(prefix)

        if not cached:
            return None

        key_hash, key_id = cached
        if not self._verify_key(raw_key, key_hash):
            return None

        # Valid - fetch full metadata
        return self.get_key_by_id(key_id)

    def get_key_by_id(self, key_id: str) -> APIKey | None:
        """Get key metadata by ID."""
        with self._lock, self._get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM api_keys WHERE id = ?", (key_id,)
            ).fetchone()
        return self._row_to_key(row) if row else None

    def get_key_by_prefix(self, prefix: str) -> APIKey | None:
        """Get key metadata by prefix."""
        with self._lock, self._get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM api_keys WHERE key_prefix = ?", (prefix,)
            ).fetchone()
        return self._row_to_key(row) if row else None

    def list_keys(
        self,
        include_revoked: bool = False,
        environment: str | None = None,
    ) -> list[APIKey]:
        """List all keys matching criteria."""
        query = "SELECT * FROM api_keys"
        params: list[str] = []
        conditions: list[str] = []

        if not include_revoked:
            conditions.append("revoked_at IS NULL")
        if environment:
            conditions.append("environment = ?")
            params.append(environment)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY created_at DESC"

        with self._lock, self._get_conn() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._row_to_key(row) for row in rows]

    def revoke_key(self, key_id: str) -> bool:
        """Revoke a key. Returns True if successful."""
        now = datetime.now(timezone.utc).isoformat()
        with self._lock, self._get_conn() as conn:
            cursor = conn.execute(
                "UPDATE api_keys SET revoked_at = ? WHERE id = ? AND revoked_at IS NULL",
                (now, key_id),
            )
            if cursor.rowcount == 0:
                return False
            # Get prefix to invalidate cache
            row = conn.execute(
                "SELECT key_prefix FROM api_keys WHERE id = ?", (key_id,)
            ).fetchone()
            if row:
                self._invalidate_cache(row["key_prefix"])
        return True

    def update_last_used(self, key_id: str) -> None:
        """Update the last_used_at timestamp for a key."""
        now = datetime.now(timezone.utc).isoformat()
        with self._lock, self._get_conn() as conn:
            conn.execute(
                "UPDATE api_keys SET last_used_at = ? WHERE id = ?",
                (now, key_id),
            )

    @staticmethod
    def _row_to_key(row: sqlite3.Row) -> APIKey:
        """Convert a database row to an APIKey model."""
        return APIKey(
            id=row["id"],
            key_prefix=row["key_prefix"],
            name=row["name"],
            environment=Environment(row["environment"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            revoked_at=(
                datetime.fromisoformat(row["revoked_at"]) if row["revoked_at"] else None
            ),
            last_used_at=(
                datetime.fromisoformat(row["last_used_at"])
                if row["last_used_at"]
                else None
            ),
            rate_limit_per_minute=row["rate_limit_per_minute"] or 60,
            expires_at=(
                datetime.fromisoformat(row["expires_at"]) if row["expires_at"] else None
            ),
        )


__all__ = ["APIKeyAdapter"]
