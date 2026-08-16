# Orchard code manifest

Indented map of the repository. Tags: `[proposed]` = planned, not yet implemented.

## Root

- `pyproject.toml` — package metadata; runtime `numpy`/`scipy`/`scikit-learn`; optional `dev` (`pytest`, `ruff`); optional `embeddings` (`torch`, `transformers`; no `sentence-transformers`)
- `README.md` — what / install / minimal example / docs link
- `CODE_MANIFEST.md` — this file (final Phase 4 map)
- `docs/` — architecture, extraction plan, release gate
- `docs/site/` — static multi-page docs site (open `index.html`; no build server)

## `src/orchard/` — installable library

- `__init__.py` — public exports
  - `Document`, `Orchard`, `OrchardBuilder`, `Tree`
  - `DomainTaxonomy`, `FunctionTaxonomy`, `TaxonomyModel`, `StubTaxonomy`
  - `label_intrinsic`, `import_labels`
  - `InvalidFusionError`, `MissingOptionalDependencyError`
  - cut / viz / identity helpers
- `document.py` — `Document`
- `identity.py` — membership hashes, SciPy index ↔ canonical IDs
- `schemas.py` — artifact / tree / cut / viz schema versions
- `exceptions.py` — `OrchardError` hierarchy; `InvalidFusionError`; `MissingOptionalDependencyError`; corpus mutation unsupported
- `tree.py` — `Tree.from_linkage`, subtree, named label storage
- `orchard.py` — multi-tree `Orchard.from_trees` / `save` / `load`
- `builder.py` — `OrchardBuilder.build` (defaults → domain+function cue/JS; `[]` → MiniLM+TF-IDF semantic or explicit TF-IDF / offline fallback)
- `taxonomy.py` — cue/fit taxonomy models + packaged defaults
- `labels.py` — `label_intrinsic`, `import_labels` (no contrastive API)
- `cuts.py` — dynamic cut, Steiner local-Z, packing, walk helpers
- `viz.py` — Plotly-oriented nested payloads
- `backends/` — TF-IDF + MiniLM + similarity/linkage + fusion engine + layer registry
  - `fusion.py` — `SimilarityProfile`, raw-convex and variance-calibrated fusion (D-027/D-030/D-031)
  - `layers.py` — layer protocol + registry (`tfidf_cosine`, `{taxonomy}_raw_js`)
  - `similarity.py` — cosine/JS + `linkage_from_similarity` / `linkage_from_dissimilarity`
  - `tfidf.py` — offline TF-IDF backend (explicit semantic opt-in)
  - `minilm.py` — revision-pinned MiniLM encode + Phase 3B transforms (D-029/D-030)
  - `modernbert.py` `[proposed]`
  - `taxonomy_heads.py` `[proposed]`
- `adapters/` — directory / JSON / JSONL / CSV / records → `Document`
  - `documents_from_*`, `load_documents`
- `assets/taxonomies/` — replaceable Domain/Function JSON + PROVENANCE
  - `heads/` `[proposed]`
- `assets/profiles/` — packaged `semantic.json` (0.66/0.34); domain/function profiles `[proposed]`
- `fixtures/` — generic reference corpus + sample linkage

### Future (not shipped)

- ModernBERT taxonomy heads `[proposed]`
- contrastive labeling `[proposed]`
- OpenAI intrinsic backend `[proposed]`

## `tests/`

- `conftest.py` — shared fixtures
- `test_identity_contracts.py` — identity / uniqueness
- `test_fixtures.py` — packaged fixtures
- `test_orchard_persist.py` — multi-tree save/load
- `test_cuts_and_subtrees.py` — partition + subtree invariants
- `test_external_linkage_and_viz.py` — external Z path + multi-tree viz
- `test_builder_semantic.py` — no-taxonomy → MiniLM fused default / TF-IDF opt-in / fallback
- `test_minilm_transforms.py` — centering / signed map / whitening / PC-3 on injected vectors; optional live encode
- `test_builder_taxonomies.py` — one tree per taxonomy
- `test_fusion.py` — fusion engine + layer registry (no GPU)
- `test_default_taxonomies.py` — Domain + Function defaults
- `test_labels_immutable_linkage.py` — label overlays immutable w.r.t. linkage
- `test_adapters.py` — adapters → `build`
- `test_readme_example.py` — README minimal example
- `test_docs_site.py` — static docs pages + sidebar coverage

## `docs/site/` pages

- Getting Started, Architecture, Documents, Taxonomies, Building Trees, Multiple Trees
- Cuts & Views, Labels, Persistence, Visualization, Adapters, Extending Orchard
