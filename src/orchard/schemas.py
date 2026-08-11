"""Schema and artifact version constants for Orchard persistence contracts."""

from __future__ import annotations

# Version of the on-disk Orchard artifact directory / manifest family.
ARTIFACT_SCHEMA_VERSION = "orchard_artifact_v1"

# Canonical document record schema.
DOCUMENT_SCHEMA_VERSION = "orchard_document_v1"

# Canonical binary linkage tree (SciPy Z + node registry).
CANONICAL_TREE_SCHEMA_VERSION = "canonical_linkage_tree_v1"

# SciPy linkage matrix artifact wrapper.
LINKAGE_SCHEMA_VERSION = "scipy_linkage_artifact_v1"

# Local linkage index ↔ canonical node ID map.
LINKAGE_INDEX_SCHEMA_VERSION = "linkage_index_map_v1"

# Nested dynamic cut over canonical nodes.
DYNAMIC_CUT_SCHEMA_VERSION = "canonical_dynamic_cut_v1"
DYNAMIC_CUTTER_SCHEMA_VERSION = "canonical_dynamic_tree_cutter_v1"

# Packed frontier presentation view.
PACKED_VIEW_SCHEMA_VERSION = "packed_canonical_view_v1"
PACKER_ADAPTER_SCHEMA_VERSION = "canonical_linkage_packer_adapter_v1"

# Plotly nested structure payload (data only; no Plotly dependency).
PLOTLY_STRUCTURE_SCHEMA_VERSION = "plotly_nested_structure_v1"
PLOTLY_ADAPTER_SCHEMA_VERSION = "orchard_plotly_nested_adapter_v1"

# Named label-set artifact.
LABEL_SET_SCHEMA_VERSION = "orchard_label_set_v1"
