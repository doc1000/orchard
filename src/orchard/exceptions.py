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


class InvalidFusionError(OrchardError, ValueError):
    """Raised when fusion weights, layers, or calibrated variance are invalid."""


class MissingOptionalDependencyError(OrchardError, ImportError):
    """Raised when a selected backend requires an uninstalled extra.

    Message must name the extra and the exact opt-in. Do not silently swap
    MiniLM for TF-IDF and still call the tree neural.
    """

    def __init__(self, extra: str, message: str | None = None) -> None:
        self.extra = extra
        text = message or (
            f"orchard[{extra}] is required. Install that extra, or opt in with "
            "allow_offline_fallback=True, embedding_backend=TfidfEmbeddingBackend(), "
            "or an explicit cue-only / custom taxonomy."
        )
        super().__init__(text)


class CorpusMutationUnsupportedError(OrchardError, NotImplementedError):
    """Raised when incremental insert/delete/mutation of a corpus is requested."""


class UnknownTreeError(OrchardError, KeyError):
    """Raised when a named tree is missing from an Orchard."""


class UnknownNodeError(OrchardError, KeyError):
    """Raised when a canonical node ID is not present in a tree."""


class UnknownLabelSetError(OrchardError, KeyError):
    """Raised when a named label set is missing."""
