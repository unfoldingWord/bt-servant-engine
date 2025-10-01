"""Identifier helpers for producing log-safe user tokens."""

from __future__ import annotations

import base64
import hmac
import hashlib
import os
from functools import lru_cache
from typing import Optional


def _encode_digest(digest: bytes) -> str:
    """URL-safe base64 encoding without padding."""
    token = base64.urlsafe_b64encode(digest).decode("ascii")
    return token.rstrip("=")


def _pseudonymize(value: str, secret: str, length: int = 16) -> str:
    """Return a deterministic pseudonym for a value using the provided secret."""
    digest = hmac.new(secret.encode("utf-8"), value.encode("utf-8"), hashlib.sha256).digest()
    token = _encode_digest(digest)
    return token[:length]


def _resolve_secret(secret: Optional[str]) -> str:
    """Return the secret to use for pseudonymization, falling back to env."""
    if secret:
        return secret
    env_secret = os.environ.get("LOG_PSEUDONYM_SECRET")
    if not env_secret:
        raise RuntimeError("LOG_PSEUDONYM_SECRET environment variable must be set.")
    return env_secret


@lru_cache(maxsize=4096)
def _pseudonym_cache(user_id: str, secret: str) -> str:
    return _pseudonymize(user_id, secret)


def get_log_safe_user_id(user_id: str, *, secret: Optional[str] = None) -> str:
    """Return a deterministic, non-reversible identifier suitable for logs."""
    resolved_secret = _resolve_secret(secret)
    return _pseudonym_cache(user_id, resolved_secret)


def clear_log_safe_user_cache() -> None:
    """Clear cached pseudonyms (useful for tests or secret rotation)."""
    _pseudonym_cache.cache_clear()
