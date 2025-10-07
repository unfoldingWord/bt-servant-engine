"""Legacy shim exposing configuration from ``bt_servant_engine.core``."""

from bt_servant_engine.core.config import Settings as Config, settings as config

__all__ = ["Config", "config"]
