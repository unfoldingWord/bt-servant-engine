"""Tests for the API key service."""

# pylint: disable=missing-function-docstring,redefined-outer-name

from datetime import datetime, timedelta, timezone

import pytest

from bt_servant_engine.adapters.api_keys import APIKeyAdapter
from bt_servant_engine.core.api_key_models import Environment
from bt_servant_engine.services.api_keys import APIKeyService


@pytest.fixture
def service(tmp_path):
    """Create a service with a temporary database."""
    db_path = tmp_path / "api_keys.db"
    adapter = APIKeyAdapter(db_path=db_path)
    return APIKeyService(adapter)


class TestAPIKeyServiceCreate:
    """Tests for creating API keys via service."""

    def test_create_key_success(self, service):
        """Should create a key successfully."""
        result = service.create_key(name="Test Key", environment="prod")

        assert result.key.name == "Test Key"
        assert result.key.environment == Environment.PROD
        assert result.raw_key.startswith("bts_prod_")

    def test_create_key_with_options(self, service):
        """Should respect rate limit and expiration."""
        expires = datetime.now(timezone.utc) + timedelta(days=7)
        result = service.create_key(
            name="Custom Key",
            environment="staging",
            rate_limit_per_minute=120,
            expires_at=expires,
        )

        assert result.key.rate_limit_per_minute == 120
        assert result.key.expires_at is not None

    def test_create_key_empty_name_raises(self, service):
        """Should raise ValueError for empty name."""
        with pytest.raises(ValueError, match="name"):
            service.create_key(name="", environment="prod")

    def test_create_key_whitespace_name_raises(self, service):
        """Should raise ValueError for whitespace-only name."""
        with pytest.raises(ValueError, match="name"):
            service.create_key(name="   ", environment="prod")

    def test_create_key_invalid_environment_raises(self, service):
        """Should raise ValueError for invalid environment."""
        with pytest.raises(ValueError, match="environment"):
            service.create_key(name="Bad Env", environment="invalid")


class TestAPIKeyServiceValidate:
    """Tests for validating API keys via service."""

    def test_validate_valid_key(self, service):
        """Should validate a correct key and update last_used."""
        result = service.create_key(name="Valid", environment="prod")

        validation = service.validate_key(result.raw_key)

        assert validation.valid is True
        assert validation.key is not None
        assert validation.key.name == "Valid"
        assert validation.error is None

    def test_validate_invalid_key(self, service):
        """Should return invalid result for wrong key."""
        validation = service.validate_key("bts_prod_notavalidkeyatall")

        assert validation.valid is False
        assert validation.key is None
        assert validation.error is not None

    def test_validate_revoked_key(self, service):
        """Should return invalid result for revoked key."""
        result = service.create_key(name="To Revoke", environment="prod")
        service.revoke_key(result.key.id)

        validation = service.validate_key(result.raw_key)

        assert validation.valid is False


class TestAPIKeyServiceList:
    """Tests for listing API keys via service."""

    def test_list_keys(self, service):
        """Should list all active keys."""
        service.create_key(name="Key 1", environment="prod")
        service.create_key(name="Key 2", environment="dev")

        keys = service.list_keys()

        assert len(keys) == 2

    def test_list_keys_filter_by_environment(self, service):
        """Should filter by environment."""
        service.create_key(name="Prod", environment="prod")
        service.create_key(name="Dev", environment="dev")

        keys = service.list_keys(environment="prod")

        assert len(keys) == 1
        assert keys[0].name == "Prod"


class TestAPIKeyServiceRevoke:
    """Tests for revoking API keys via service."""

    def test_revoke_key_by_id(self, service):
        """Should revoke key by ID."""
        result = service.create_key(name="To Revoke", environment="prod")

        success = service.revoke_key(result.key.id)

        assert success is True
        keys = service.list_keys()
        assert len(keys) == 0

    def test_revoke_key_by_prefix(self, service):
        """Should revoke key by prefix."""
        result = service.create_key(name="By Prefix", environment="prod")

        success = service.revoke_key_by_prefix(result.key.key_prefix)

        assert success is True
        keys = service.list_keys()
        assert len(keys) == 0

    def test_revoke_nonexistent_key(self, service):
        """Should return False for nonexistent key."""
        success = service.revoke_key("nonexistent")

        assert success is False
