# Orchard code manifest

Indented map of the repository. Tags: `[proposed]` = planned, not yet implemented.

## Root

- `pyproject.toml` — package metadata, runtime deps (`numpy`), optional `dev` (`pytest`)
- `README.md` — short package identity (expanded in Phase 4)
- `CODE_MANIFEST.md` — this file; updated every phase
- `docs/` — architecture, extraction plan, release gate (not the product docs site yet)

## `src/orchard/` — installable library

- `__init__.py` — package version and Phase 0 public contract exports
  - `__version__`, `Document`, identity helpers, exception types, schema constants
- `document.py` — canonical corpus record
  - `Document`
- `identity.py` — item / tree / canonical node IDs; SciPy index ↔ node ID map
  - `membership_hash`, `canonical_node_id`, `generate_item_id`, `validate_tree_id`
  - `ensure_unique_item_ids`, `assign_canonical_ids`, `LinkageIdentity`
  - `leaf_scipy_index`, `internal_scipy_index`, `root_scipy_index`
- `schemas.py` — schema / artifact version strings
  - `ARTIFACT_SCHEMA_VERSION`, `DOCUMENT_SCHEMA_VERSION`, `CANONICAL_TREE_SCHEMA_VERSION`
  - `LINKAGE_SCHEMA_VERSION`, `LINKAGE_INDEX_SCHEMA_VERSION`
- `exceptions.py` — shared error types
  - `OrchardError`, `DuplicateItemIdError`, `InvalidIdentityError`
  - `InvalidLinkageError`, `CorpusMutationUnsupportedError`
- `fixtures/` — generic reference corpus + sample linkage (no AppWorld)
  - `load_documents`, `load_sample_linkage`, `load_sample_linkage_identity`
  - `documents.json`, `sample_linkage.json`

### Proposed (later phases)

- `orchard.py` `[proposed]` — multi-tree `Orchard` owner; `from_trees`, save/load
- `tree.py` `[proposed]` — `Tree.from_linkage`; registry; named label storage
- `builder.py` `[proposed]` — `OrchardBuilder.build(documents)` taxonomy/semantic branching
- `cuts.py` `[proposed]` — dynamic cut, walk-cut, Steiner subtree, packing
- `viz.py` `[proposed]` — Plotly nested payloads; multi-tree selection
- `labels.py` `[proposed]` — intrinsic / imported / named label sets (no contrastive)
- `taxonomy.py` `[proposed]` — taxonomy load/transform; Domain/Function defaults
- `adapters/` `[proposed]` — directory / JSON / JSONL / CSV → `Document`
- `backends/` `[proposed]` — embedding / similarity backends cleaned from demo bridges

## `tests/`

- `test_identity_contracts.py` — Document defaults, unique IDs, node IDs, linkage map
- `test_fixtures.py` — packaged fixtures load and match identity contracts

### Proposed tests

- `test_orchard_persist.py` `[proposed]` — multi-tree save/load identity
- `test_cuts_and_subtrees.py` `[proposed]` — partition + recursive subtree invariants
- `test_builder_semantic.py` `[proposed]` — no-taxonomy → `semantic` tree
- `test_builder_taxonomies.py` `[proposed]` — one tree per taxonomy
- `test_labels_immutable_linkage.py` `[proposed]` — label swap never mutates Z
- `test_readme_example.py` `[proposed]` — README minimal example
