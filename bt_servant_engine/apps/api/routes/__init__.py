"""Router namespace exports for FastAPI include hooks."""

from . import admin_datastore, admin_logs, admin_status_messages, health, webhooks

__all__ = ["admin_datastore", "admin_logs", "admin_status_messages", "health", "webhooks"]
