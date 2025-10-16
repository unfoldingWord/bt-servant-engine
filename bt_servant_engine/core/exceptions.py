"""Core exception types shared across layers."""


class CollectionExistsError(Exception):
    """Raised when attempting to create a collection that already exists."""


class CollectionNotFoundError(Exception):
    """Raised when attempting to access or delete a missing collection."""


class DocumentNotFoundError(Exception):
    """Raised when attempting to delete a missing document from a collection."""


__all__ = [
    "CollectionExistsError",
    "CollectionNotFoundError",
    "DocumentNotFoundError",
]
