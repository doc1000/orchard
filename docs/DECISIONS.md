# Orchard decisions

Durable product decisions for the standalone Orchard library. AppWorld
evaluation notes stay in `tool-tree-demo/docs/`. This file supersedes
extraction-plan non-goals where they conflict.

## D-O-001 — Fused per-tree profiles are first-class

Each named tree (`domain`, `function`, `semantic`) is built from independent
similarity layers, an inspectable `SimilarityProfile`, fusion, dissimilarity,
and average linkage.

The extraction-plan non-goal “fused default trees” is **reversed for per-tree
fusion**. It meant “do not ship one mixed Domain+Function+semantic orchard
tree,” not “each named tree may use only one layer.”

There is still **no** orchard tree named `fused` or `mixed`.

## D-O-002 — Independent matrices (D-002)

Layers are `[n, n]` similarity matrices. Do not concatenate heterogeneous
feature vectors as the primary fusion method. Do not silently renormalize
weights (`abs_tol=1e-12`). Missing layers are absent; ship the matching dict.

## D-O-003 — MiniLM is the description embedding

Default description layer is revision-pinned MiniLM
(`sentence-transformers/all-MiniLM-L6-v2`), corpus-centered, then signed
cosine. ModernBERT is **not** the description embedding. It is the taxonomy
feature encoder for packaged logistic heads only.

Semantic no-app pair stays `description_minilm_centered_cosine=0.66` /
`tfidf_cosine=0.34`.

## D-O-004 — Variance-calibrated default

`fusion_mode` default is `variance_calibrated`. `raw_convex` (`D = 1 − S`)
remains available. Ward is rejected on this precomputed-dissimilarity path.
Variance-calibrated fusion needs n≥3; a selected layer with off-diagonal
variance ≤ `1e-12` aborts.

## D-O-005 — No-app weight dicts

Documents usually have no `app_name`. Default profiles omit `app_exact_match`
and fold that 0.03 into the dominant taxonomy JS. Packaged dicts (already sum
to 1):

| Tree | When | Weights |
|---|---|---|
| `semantic` | always, no app | MiniLM `0.66`, TF-IDF `0.34` |
| `domain` | both taxonomies | `domain_raw_js=0.48`, MiniLM `0.23`, TF-IDF `0.14`, `function_raw_js=0.15` |
| `function` | both taxonomies | `function_raw_js=0.48`, MiniLM `0.25`, TF-IDF `0.15`, `domain_raw_js=0.12` |
| `domain` | only domain | `domain_raw_js=0.63`, MiniLM `0.23`, TF-IDF `0.14` |
| `function` | only function | `function_raw_js=0.60`, MiniLM `0.25`, TF-IDF `0.15` |

`function_raw_js` is AppWorld `functional_raw_js`. With-app `0.03` dicts are
Phase 5.

## D-O-006 — Extras policy (no silent fallback)

Neural backends are the intended defaults. Missing a required extra without
opt-in raises `MissingOptionalDependencyError` naming the extra and the
opt-in. Do not drop MiniLM or a JS layer and renormalize.

- Default `OrchardBuilder()` needs `orchard[taxonomy-ml]` **and**
  `orchard[embeddings]` (or injected encoder + MiniLM).
- `allow_offline_fallback=True` without extras → single-layer trees
  (`tfidf_cosine=1.0` / own-taxonomy JS `1.0`), provenance
  `offline_fallback`, `taxonomy_transform: "cue"`.
- Explicit `TfidfEmbeddingBackend()`, cue-only / custom taxonomies, or an
  explicit `SimilarityProfile` remain valid opt-ins.

## D-O-007 — Packaged Phase 2B heads are the taxonomy-ml default

When `orchard[taxonomy-ml]` (or an injected classifier backend) is present,
default Domain/Function load packaged student heads.
`taxonomy_transform` is `modernbert_logistic`. Cue is the explicit fallback,
not the neural default. Heads are an AppWorld-tool prior, not a universal
document classifier. See `src/orchard/assets/taxonomies/heads/PROVENANCE.md`.
