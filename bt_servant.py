"""Top-level FastAPI entrypoint compatible with legacy imports."""

from bt_servant_engine.apps.api.app import create_app

app = create_app()


def init() -> None:
    """Deprecated startup hook retained for test compatibility."""
    return None


__all__ = ["app", "init", "create_app"]
