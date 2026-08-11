# Orchard code manifest

Indented map of the repository. Tags: `[proposed]` = planned, not yet implemented.

## Root

- `pyproject.toml` — package metadata; runtime deps `numpy`, `scipy`, `scikit-learn`; optional `dev` (`pytest`)
- `README.md` — short package identity (expanded in Phase 4)
- `CODE_MANIFEST.md` — this file; updated every phase
- `docs/` — architecture, extraction plan, release gate (not the product docs site yet)

## `src/orchard/` — installable library

- `__init__.py` — package version and public exports
  - `__version__`, `Document`, `Orchard`, `OrchardBuilder`, `Tree`
  - `DomainTaxonomy`, `FunctionTaxonomy`, `TaxonomyModel`, `StubTaxonomy`
  - `label_intrinsic`, `import_labels` (no contrastive API)
- `document.py` — canonical corpus record
- `identity.py` — item / tree / canonical node IDs; SciPy index ↔ node ID map
- `schemas.py` — schema / artifact version strings
- `exceptions.py` — shared error types
- `tree.py` — canonical binary tree; named label storage overlays
- `orchard.py` — multi-tree owner; save/load
- `builder.py` — `OrchardBuilder.build`
  - default taxonomies → Domain + Function; `taxonomies=[]` → `semantic`
- `taxonomy.py` — taxonomy protocol + cue/fit models + defaults
  - `Taxonomy`, `TaxonomyModel`, `StubTaxonomy`, `DomainTaxonomy`, `FunctionTaxonomy`
  - `default_taxonomies`, `TaxonomyModel.fit` / `save` / `load` / `transform`
- `labels.py` — intrinsic heuristic + imported overlays
  - `label_intrinsic`, `import_labels`
- `cuts.py` — dynamic cut, Steiner local-Z, packing, walk helpers
- `viz.py` — Plotly-oriented nested payloads
- `backends/` — offline TF-IDF + similarity/linkage cores
- `assets/taxonomies/` — replaceable Domain/Function definition JSON + PROVENANCE
  - `domain.json`, `function.json`
- `fixtures/` — generic reference corpus + sample linkage

### Proposed (later phases)

- `adapters/` `[proposed]` — directory / JSON / JSONL / CSV → `Document`
- `backends/sentence_transformers.py` `[proposed]` — optional neural embedding backend
- contrastive labeling `[proposed]` — future only; not in public API

## `tests/`

- `conftest.py` — shared document/tree/orchard fixtures
- `test_identity_contracts.py` — Document defaults, unique IDs, node IDs, linkage map
- `test_fixtures.py` — packaged fixtures load and match identity contracts
- `test_orchard_persist.py` — multi-tree save/load identity
- `test_cuts_and_subtrees.py` — partition + recursive subtree invariants
- `test_external_linkage_and_viz.py` — external Z cut/plot/persist/labels
- `test_builder_semantic.py` — no-taxonomy → `semantic` tree (offline)
- `test_builder_taxonomies.py` — one tree per taxonomy; no fused default
- `test_default_taxonomies.py` — Domain + Function defaults; replaceable artifacts
- `test_labels_immutable_linkage.py` — intrinsic/imported labels; no contrastive API

### Proposed tests

- `test_readme_example.py` `[proposed]` — README minimal example
- adapter / docs-site tests `[proposed]` — Phase 4
