"""Orchard: persistent hierarchical trees over a fixed document corpus."""

from orchard.builder import OrchardBuilder
from orchard.cuts import (
    build_dynamic_cut,
    cut_partition_item_ids,
    linkage_for_canonical_subtree,
    pack_canonical_tree,
    validate_dynamic_cut,
    walk_cut_json,
)
from orchard.document import Document
from orchard.exceptions import (
    CorpusMutationUnsupportedError,
    DuplicateItemIdError,
    InvalidFusionError,
    InvalidIdentityError,
    InvalidLinkageError,
    OrchardError,
    UnknownLabelSetError,
    UnknownNodeError,
    UnknownTreeError,
)
from orchard.identity import (
    assign_canonical_ids,
    canonical_node_id,
    ensure_unique_item_ids,
    generate_item_id,
    membership_hash,
    validate_tree_id,
)
from orchard.labels import import_labels, label_intrinsic
from orchard.orchard import Orchard
from orchard.schemas import ARTIFACT_SCHEMA_VERSION, DOCUMENT_SCHEMA_VERSION
from orchard.taxonomy import (
    DomainTaxonomy,
    FunctionTaxonomy,
    StubTaxonomy,
    TaxonomyModel,
    default_taxonomies,
)
from orchard.tree import Tree
from orchard.viz import (
    cut_to_plotly_structure,
    orchard_plotly_structures,
    validate_plotly_structure,
)

__version__ = "0.1.0"

__all__ = [
    "ARTIFACT_SCHEMA_VERSION",
    "DOCUMENT_SCHEMA_VERSION",
    "CorpusMutationUnsupportedError",
    "Document",
    "DomainTaxonomy",
    "DuplicateItemIdError",
    "FunctionTaxonomy",
    "InvalidFusionError",
    "InvalidIdentityError",
    "InvalidLinkageError",
    "Orchard",
    "OrchardBuilder",
    "OrchardError",
    "StubTaxonomy",
    "TaxonomyModel",
    "Tree",
    "UnknownLabelSetError",
    "UnknownNodeError",
    "UnknownTreeError",
    "assign_canonical_ids",
    "build_dynamic_cut",
    "canonical_node_id",
    "cut_partition_item_ids",
    "cut_to_plotly_structure",
    "default_taxonomies",
    "ensure_unique_item_ids",
    "generate_item_id",
    "import_labels",
    "label_intrinsic",
    "linkage_for_canonical_subtree",
    "membership_hash",
    "orchard_plotly_structures",
    "pack_canonical_tree",
    "validate_dynamic_cut",
    "validate_plotly_structure",
    "validate_tree_id",
    "walk_cut_json",
    "__version__",
]
