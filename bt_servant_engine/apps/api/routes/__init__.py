"""Router namespace exports for FastAPI include hooks."""

from . import admin_datastore, admin_logs, health, webhooks

__all__ = ["admin_datastore", "admin_logs", "health", "webhooks"]
