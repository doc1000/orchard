"""Orchard exception types."""

from __future__ import annotations


class OrchardError(Exception):
    """Base error for Orchard contracts and operations."""


class DuplicateItemIdError(OrchardError, ValueError):
    """Raised when a corpus contains duplicate item IDs."""


class InvalidIdentityError(OrchardError, ValueError):
    """Raised when item, tree, or node identity contracts are violated."""


class InvalidLinkageError(OrchardError, ValueError):
    """Raised when a linkage matrix cannot be mapped to canonical IDs."""


class CorpusMutationUnsupportedError(OrchardError, NotImplementedError):
    """Raised when incremental insert/delete/mutation of a corpus is requested."""
