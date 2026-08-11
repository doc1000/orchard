"""Orchard: persistent hierarchical trees over a fixed document corpus."""

from orchard.document import Document
from orchard.exceptions import (
    CorpusMutationUnsupportedError,
    DuplicateItemIdError,
    InvalidIdentityError,
    InvalidLinkageError,
    OrchardError,
)
from orchard.identity import (
    assign_canonical_ids,
    canonical_node_id,
    ensure_unique_item_ids,
    generate_item_id,
    membership_hash,
    validate_tree_id,
)
from orchard.schemas import ARTIFACT_SCHEMA_VERSION, DOCUMENT_SCHEMA_VERSION

__version__ = "0.1.0"

__all__ = [
    "ARTIFACT_SCHEMA_VERSION",
    "DOCUMENT_SCHEMA_VERSION",
    "CorpusMutationUnsupportedError",
    "Document",
    "DuplicateItemIdError",
    "InvalidIdentityError",
    "InvalidLinkageError",
    "OrchardError",
    "assign_canonical_ids",
    "canonical_node_id",
    "ensure_unique_item_ids",
    "generate_item_id",
    "membership_hash",
    "validate_tree_id",
    "__version__",
]
