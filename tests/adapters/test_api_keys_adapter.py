"""Tests for the API key SQLite adapter."""

# pylint: disable=missing-function-docstring,redefined-outer-name

from datetime import datetime, timedelta, timezone

import pytest

from bt_servant_engine.adapters.api_keys import APIKeyAdapter
from bt_servant_engine.core.api_key_models import Environment


@pytest.fixture
def adapter(tmp_path):
    """Create an adapter with a temporary database."""
    db_path = tmp_path / "api_keys.db"
    return APIKeyAdapter(db_path=db_path)


class TestAPIKeyAdapterCreate:
    """Tests for creating API keys."""

    def test_create_key_returns_key_and_raw_key(self, adapter):
        """Should return an APIKey and the raw key string."""
        key, raw = adapter.create_key(name="Test Key", environment="prod")

        assert key.name == "Test Key"
        assert key.environment == Environment.PROD
        assert key.is_active is True
        assert raw.startswith("bts_prod_")
        assert len(raw) > 20

    def test_create_key_with_rate_limit(self, adapter):
        """Should respect custom rate limit."""
        key, _ = adapter.create_key(
            name="Limited Key", environment="staging", rate_limit_per_minute=100
        )

        assert key.rate_limit_per_minute == 100

    def test_create_key_with_expiration(self, adapter):
        """Should set expiration date."""
        expires = datetime.now(timezone.utc) + timedelta(days=30)
        key, _ = adapter.create_key(name="Expiring Key", environment="dev", expires_at=expires)

        assert key.expires_at is not None
        assert abs((key.expires_at - expires).total_seconds()) < 1

    def test_create_key_prefix_is_unique(self, adapter):
        """Each key should have a unique prefix."""
        key1, _ = adapter.create_key(name="Key 1", environment="prod")
        key2, _ = adapter.create_key(name="Key 2", environment="prod")

        assert key1.key_prefix != key2.key_prefix


class TestAPIKeyAdapterValidate:
    """Tests for validating API keys."""

    def test_validate_valid_key(self, adapter):
        """Should validate a correct key."""
        _, raw = adapter.create_key(name="Valid Key", environment="prod")

        result = adapter.validate_key(raw)

        assert result is not None
        assert result.name == "Valid Key"

    def test_validate_invalid_key(self, adapter):
        """Should return None for invalid key."""
        result = adapter.validate_key("bts_prod_invalidkeynotindb")

        assert result is None

    def test_validate_wrong_format(self, adapter):
        """Should return None for malformed key."""
        result = adapter.validate_key("not-a-valid-key-format")

        assert result is None

    def test_validate_revoked_key(self, adapter):
        """Should return None for revoked key."""
        key, raw = adapter.create_key(name="To Revoke", environment="prod")
        adapter.revoke_key(key.id)

        result = adapter.validate_key(raw)

        assert result is None

    def test_validate_expired_key(self, adapter):
        """Adapter returns key even if expired - service layer checks is_active."""
        expires = datetime.now(timezone.utc) - timedelta(days=1)
        _, raw = adapter.create_key(name="Expired Key", environment="prod", expires_at=expires)

        result = adapter.validate_key(raw)

        # Adapter returns the key - it's the service's job to check is_active
        assert result is not None
        assert result.is_active is False  # But the key reports itself as inactive


class TestAPIKeyAdapterList:
    """Tests for listing API keys."""

    def test_list_keys_empty(self, adapter):
        """Should return empty list when no keys."""
        keys = adapter.list_keys()

        assert keys == []

    def test_list_keys_returns_all_active(self, adapter):
        """Should return all active keys."""
        adapter.create_key(name="Key 1", environment="prod")
        adapter.create_key(name="Key 2", environment="staging")

        keys = adapter.list_keys()

        assert len(keys) == 2
        names = {k.name for k in keys}
        assert names == {"Key 1", "Key 2"}

    def test_list_keys_excludes_revoked_by_default(self, adapter):
        """Should not include revoked keys by default."""
        key, _ = adapter.create_key(name="Active", environment="prod")
        key2, _ = adapter.create_key(name="Revoked", environment="prod")
        adapter.revoke_key(key2.id)

        keys = adapter.list_keys()

        assert len(keys) == 1
        assert keys[0].name == "Active"

    def test_list_keys_includes_revoked_when_requested(self, adapter):
        """Should include revoked keys when requested."""
        adapter.create_key(name="Active", environment="prod")
        key2, _ = adapter.create_key(name="Revoked", environment="prod")
        adapter.revoke_key(key2.id)

        keys = adapter.list_keys(include_revoked=True)

        assert len(keys) == 2

    def test_list_keys_filter_by_environment(self, adapter):
        """Should filter by environment."""
        adapter.create_key(name="Prod Key", environment="prod")
        adapter.create_key(name="Dev Key", environment="dev")

        prod_keys = adapter.list_keys(environment="prod")
        dev_keys = adapter.list_keys(environment="dev")

        assert len(prod_keys) == 1
        assert prod_keys[0].name == "Prod Key"
        assert len(dev_keys) == 1
        assert dev_keys[0].name == "Dev Key"


class TestAPIKeyAdapterRevoke:
    """Tests for revoking API keys."""

    def test_revoke_existing_key(self, adapter):
        """Should revoke an existing key."""
        key, _ = adapter.create_key(name="To Revoke", environment="prod")

        result = adapter.revoke_key(key.id)

        assert result is True
        keys = adapter.list_keys(include_revoked=True)
        assert keys[0].is_active is False

    def test_revoke_nonexistent_key(self, adapter):
        """Should return False for nonexistent key."""
        result = adapter.revoke_key("nonexistent-id")

        assert result is False

    def test_revoke_already_revoked_key(self, adapter):
        """Should return False for already revoked key."""
        key, _ = adapter.create_key(name="Already Revoked", environment="prod")
        adapter.revoke_key(key.id)

        result = adapter.revoke_key(key.id)

        assert result is False


class TestAPIKeyAdapterGetters:
    """Tests for get_key_by_id and get_key_by_prefix."""

    def test_get_key_by_id(self, adapter):
        """Should retrieve key by ID."""
        key, _ = adapter.create_key(name="By ID", environment="prod")

        result = adapter.get_key_by_id(key.id)

        assert result is not None
        assert result.id == key.id
        assert result.name == "By ID"

    def test_get_key_by_id_not_found(self, adapter):
        """Should return None for nonexistent ID."""
        result = adapter.get_key_by_id("nonexistent-id")

        assert result is None

    def test_get_key_by_prefix(self, adapter):
        """Should retrieve key by prefix."""
        key, _ = adapter.create_key(name="By Prefix", environment="prod")

        result = adapter.get_key_by_prefix(key.key_prefix)

        assert result is not None
        assert result.key_prefix == key.key_prefix

    def test_get_key_by_prefix_not_found(self, adapter):
        """Should return None for nonexistent prefix."""
        result = adapter.get_key_by_prefix("bts_prod_xxxx")

        assert result is None


class TestAPIKeyAdapterLastUsed:
    """Tests for update_last_used."""

    def test_update_last_used(self, adapter):
        """Should update last_used_at timestamp."""
        key, _ = adapter.create_key(name="Track Usage", environment="prod")
        assert key.last_used_at is None

        adapter.update_last_used(key.id)

        updated = adapter.get_key_by_id(key.id)
        assert updated is not None
        assert updated.last_used_at is not None
