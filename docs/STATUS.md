# Orchard status

Fused-similarity work from `docs/FUSED_SIMILARITY_PLAN.md`.

| Phase | Scope | Status |
|---|---|---|
| 1 | Fusion engine + layer registry; `SimilarityProfile`; `linkage_from_dissimilarity`; reject `ward` | Done |
| 2 | MiniLM semantic default (`0.66` / `0.34`, `variance_calibrated`); TF-IDF and offline opt-ins | Done |
| 3 | Packaged Domain/Function ModernBERT+logistic heads; cue is explicit fallback | Done |
| 4 | Wire fused taxonomy trees + exposed config; shared layers; docs | Done |
| 5 | App-family layer / with-app `0.03` dicts; diagnostic MiniLM flags; hardening | **Done** |

Phase 5 default: no family key → Phase 4 no-app fused dicts. Set
`family_metadata_key` and supply a non-empty value on every document →
domain/function use the G3 with-app dicts; `app_exact_match` is computed
once. Partial family metadata is a loud error. Semantic stays MiniLM/TF-IDF
`0.66` / `0.34`. `description_transform` default remains `centered`. No tree
named `fused` / `mixed`.
