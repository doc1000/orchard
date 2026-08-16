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
D-O-008.

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

## D-O-008 — App-family layer is opt-in; partial metadata is a loud error

`app_exact_match` is enabled only when **both** are true:

1. every document has a non-empty `metadata[family_metadata_key]` value, and
2. the caller set `family_metadata_key` **or** a selected `SimilarityProfile`
   includes `app_exact_match`.

Default `family_metadata_key` is `None`. Presence of `app_name` on some
documents does **not** auto-enable the layer. Offline fallback, explicit
TF-IDF, and cue-only single-layer paths do not auto-enable family.

Partial or empty family metadata raises `InvalidFusionError` naming the
missing item ids. Do not drop the layer and renormalize.

When the layer is actually in a tree’s weights, packaged **with-app** dicts
(AppWorld G3; `keyword_tfidf_cosine` → `tfidf_cosine`) are:

| Tree | When | Weights |
|---|---|---|
| `domain` | both taxonomies + app | `app_exact_match=0.03`, MiniLM `0.23`, `domain_raw_js=0.45`, `function_raw_js=0.15`, `tfidf_cosine=0.14` |
| `function` | both taxonomies + app | `app_exact_match=0.03`, MiniLM `0.25`, `domain_raw_js=0.12`, `function_raw_js=0.45`, `tfidf_cosine=0.15` |
| `domain` | only domain + app | `domain_raw_js=0.60`, MiniLM `0.23`, TF-IDF `0.14`, app `0.03` |
| `function` | only function + app | `function_raw_js=0.57`, MiniLM `0.25`, TF-IDF `0.15`, app `0.03` |

Semantic stays MiniLM `0.66` / TF-IDF `0.34` even when the family layer is
on. There is no packaged semantic-with-app default. Trees stay named
`domain` / `function` / `semantic`.

`description_transform` default remains `centered`. `raw`, `whitened`, and
`centered_pc3` are diagnostic overrides (D-029); they remap the MiniLM
layer name and are not defaults.

Optional dense-matrix persist policy is `always | never | below_size_limit |
compressed`. Default is `never`.

## AppWorld D-027–D-031 mapping

Orchard ports the numerics; it does not import `tool_tree_demo`. Layer name
`function_raw_js` is AppWorld `functional_raw_js`.

| Decision | AppWorld source | Orchard module / function |
|---|---|---|
| D-027 independent matrices + convex fusion + exact-app | `phase3.py` `app_matrix`, `fuse`, `load_profiles` | `backends/family.py` `AppExactMatchLayer`; `backends/fusion.py` `validate_fusion_weights`, `raw_convex_fusion`; `backends/layers.py` |
| D-028 Phase 3 / G3 completion boundary | Phase 3 packet / gate (no runtime formula) | Documented only; no Orchard runtime module |
| D-029 MiniLM description recalibration | `phase3b.py` `build_description_variants`, `_raw_cosine`, `_mapped_cosine`; `sentence_embedding_bridge.py` | `backends/minilm.py` `apply_description_transform`, `raw_cosine`, `mapped_cosine`, `MiniLMEmbeddingBackend` |
| D-030 reduced profiles + variance-calibrated fusion | `phase3b.py` `variance_calibrated_fusion`; `phase3b_reduced_profiles.v1.json` | `backends/fusion.py` `variance_calibrated_fusion`; `assets/profiles/{semantic,domain,function}.json` |
| D-031 G3 hierarchy: `D=1-S` / consume calibrated D; average linkage; no Ward | `phase4a.py` `_dissimilarity` | `backends/similarity.py` `linkage_from_dissimilarity`; `backends/fusion.py` `fuse_to_dissimilarity` |
