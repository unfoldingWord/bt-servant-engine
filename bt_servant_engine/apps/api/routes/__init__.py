"""Router namespace exports for FastAPI include hooks."""

from . import (
    admin_datastore,
    admin_keys,
    admin_logs,
    admin_status_messages,
    chat,
    health,
    users,
)

__all__ = [
    "admin_datastore",
    "admin_keys",
    "admin_logs",
    "admin_status_messages",
    "chat",
    "health",
    "users",
]
