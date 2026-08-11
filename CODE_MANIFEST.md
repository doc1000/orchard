# Orchard code manifest

Indented map of the repository. Tags: `[proposed]` = planned, not yet implemented.

## Root

- `pyproject.toml` — package metadata; runtime deps `numpy`, `scipy`, `scikit-learn`; optional `dev` (`pytest`)
- `README.md` — short package identity (expanded in Phase 4)
- `CODE_MANIFEST.md` — this file; updated every phase
- `docs/` — architecture, extraction plan, release gate (not the product docs site yet)

## `src/orchard/` — installable library

- `__init__.py` — package version and public exports
  - `__version__`, `Document`, `Orchard`, `OrchardBuilder`, `Tree`, `StubTaxonomy`
  - cut/viz helpers, identity helpers, exceptions
- `document.py` — canonical corpus record
  - `Document`
- `identity.py` — item / tree / canonical node IDs; SciPy index ↔ node ID map
  - `membership_hash`, `canonical_node_id`, `generate_item_id`, `validate_tree_id`
  - `ensure_unique_item_ids`, `assign_canonical_ids`, `LinkageIdentity`
- `schemas.py` — schema / artifact version strings
- `exceptions.py` — shared error types
- `tree.py` — canonical binary tree from external or built linkage
  - `Tree`, `Tree.from_linkage`, `Tree.subtree`, `Tree.set_labels`, `Tree.use_labels`
  - `build_canonical_nodes`, `validate_canonical_tree`
- `orchard.py` — multi-tree owner over shared documents
  - `Orchard`, `Orchard.from_trees`, `Orchard.tree`, `Orchard.save`, `Orchard.load`
- `builder.py` — public `build` construction path
  - `OrchardBuilder`, `OrchardBuilder.build`, `OrchardBuilder.get_params`, `normalize_documents`
  - branching: no taxonomies → `semantic`; taxonomies → one tree per taxonomy name
- `taxonomy.py` — taxonomy protocol + fixture stub (defaults in Phase 3)
  - `Taxonomy` protocol, `StubTaxonomy`
- `cuts.py` — dynamic cut, Steiner local-Z, packing, walk helpers
- `viz.py` — Plotly-oriented nested payloads (no Plotly dependency)
- `backends/` — offline feature/similarity numeric cores
  - `tfidf.py` — `TfidfEmbeddingBackend`, `tfidf_matrix`
  - `similarity.py` — `cosine_matrix`, `jensen_shannon_matrix`, `linkage_from_similarity`
- `fixtures/` — generic reference corpus + sample linkage (no AppWorld)

### Proposed (later phases)

- `labels.py` `[proposed]` — intrinsic labeling backends (named sets already on `Tree`)
- `taxonomy.py` defaults `[proposed]` — Domain/Function trained artifacts (stubs exist)
- `adapters/` `[proposed]` — directory / JSON / JSONL / CSV → `Document`
- `backends/sentence_transformers.py` `[proposed]` — optional neural embedding backend

## `tests/`

- `conftest.py` — shared document/tree/orchard fixtures
- `test_identity_contracts.py` — Document defaults, unique IDs, node IDs, linkage map
- `test_fixtures.py` — packaged fixtures load and match identity contracts
- `test_orchard_persist.py` — multi-tree save/load identity
- `test_cuts_and_subtrees.py` — partition + recursive subtree invariants
- `test_external_linkage_and_viz.py` — external Z cut/plot/persist/labels; multi-tree viz
- `test_builder_semantic.py` — no-taxonomy → `semantic` tree (offline)
- `test_builder_taxonomies.py` — one tree per taxonomy; no fused default

### Proposed tests

- `test_labels_immutable_linkage.py` `[proposed]` — label swap never mutates Z (partial coverage in Phase 1)
- `test_default_taxonomies.py` `[proposed]` — Domain + Function default artifacts
- `test_readme_example.py` `[proposed]` — README minimal example
