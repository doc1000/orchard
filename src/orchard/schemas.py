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
