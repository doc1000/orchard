# Orchard code manifest

Indented map of the repository. Tags: `[proposed]` = planned, not yet implemented.

## Root

- `pyproject.toml` — package metadata; runtime deps `numpy`, `scipy`, `scikit-learn`; optional `dev` (`pytest`)
- `README.md` — short package identity (expanded in Phase 4)
- `CODE_MANIFEST.md` — this file; updated every phase
- `docs/` — architecture, extraction plan, release gate (not the product docs site yet)

## `src/orchard/` — installable library

- `__init__.py` — package version and public exports
  - `__version__`, `Document`, `Orchard`, `Tree`, cut/viz helpers, identity helpers, exceptions
- `document.py` — canonical corpus record
  - `Document`
- `identity.py` — item / tree / canonical node IDs; SciPy index ↔ node ID map
  - `membership_hash`, `canonical_node_id`, `generate_item_id`, `validate_tree_id`
  - `ensure_unique_item_ids`, `assign_canonical_ids`, `LinkageIdentity`
  - `leaf_scipy_index`, `internal_scipy_index`, `root_scipy_index`
- `schemas.py` — schema / artifact version strings
  - `ARTIFACT_SCHEMA_VERSION`, `DOCUMENT_SCHEMA_VERSION`, `CANONICAL_TREE_SCHEMA_VERSION`
  - `LINKAGE_SCHEMA_VERSION`, `LINKAGE_INDEX_SCHEMA_VERSION`
  - `DYNAMIC_CUT_SCHEMA_VERSION`, `PACKED_VIEW_SCHEMA_VERSION`
  - `PLOTLY_STRUCTURE_SCHEMA_VERSION`, `LABEL_SET_SCHEMA_VERSION`
- `exceptions.py` — shared error types
  - `OrchardError`, `DuplicateItemIdError`, `InvalidIdentityError`, `InvalidLinkageError`
  - `CorpusMutationUnsupportedError`, `UnknownTreeError`, `UnknownNodeError`, `UnknownLabelSetError`
- `tree.py` — canonical binary tree from external linkage
  - `Tree`, `Tree.from_linkage`, `Tree.subtree`, `Tree.set_labels`, `Tree.use_labels`
  - `build_canonical_nodes`, `validate_canonical_tree`, `default_node_label`
- `orchard.py` — multi-tree owner over shared documents
  - `Orchard`, `Orchard.from_trees`, `Orchard.tree`, `Orchard.save`, `Orchard.load`
- `cuts.py` — dynamic cut, Steiner local-Z, packing, walk helpers
  - `build_dynamic_cut`, `validate_dynamic_cut`, `get_optimal_cut_labels`
  - `linkage_for_canonical_subtree`, `pack_canonical_tree`
  - `cut_partition_item_ids`, `walk_cut_json`
- `viz.py` — Plotly-oriented nested payloads (no Plotly dependency)
  - `cut_to_plotly_structure`, `orchard_plotly_structures`, `validate_plotly_structure`
- `fixtures/` — generic reference corpus + sample linkage (no AppWorld)
  - `load_documents`, `load_sample_linkage`, `load_sample_linkage_identity`
  - `documents.json`, `sample_linkage.json`

### Proposed (later phases)

- `builder.py` `[proposed]` — `OrchardBuilder.build(documents)` taxonomy/semantic branching
- `labels.py` `[proposed]` — intrinsic labeling backends (named sets already on `Tree`)
- `taxonomy.py` `[proposed]` — taxonomy load/transform; Domain/Function defaults
- `adapters/` `[proposed]` — directory / JSON / JSONL / CSV → `Document`
- `backends/` `[proposed]` — embedding / similarity backends cleaned from demo bridges

## `tests/`

- `conftest.py` — shared document/tree/orchard fixtures
- `test_identity_contracts.py` — Document defaults, unique IDs, node IDs, linkage map
- `test_fixtures.py` — packaged fixtures load and match identity contracts
- `test_orchard_persist.py` — multi-tree save/load identity
- `test_cuts_and_subtrees.py` — partition + recursive subtree invariants
- `test_external_linkage_and_viz.py` — external Z cut/plot/persist/labels; multi-tree viz

### Proposed tests

- `test_builder_semantic.py` `[proposed]` — no-taxonomy → `semantic` tree
- `test_builder_taxonomies.py` `[proposed]` — one tree per taxonomy
- `test_labels_immutable_linkage.py` `[proposed]` — label swap never mutates Z (partial coverage in Phase 1)
- `test_readme_example.py` `[proposed]` — README minimal example
