# Orchard status

Fused-similarity work from `docs/FUSED_SIMILARITY_PLAN.md`.

| Phase | Scope | Status |
|---|---|---|
| 1 | Fusion engine + layer registry; `SimilarityProfile`; `linkage_from_dissimilarity`; reject `ward` | Done |
| 2 | MiniLM semantic default (`0.66` / `0.34`, `variance_calibrated`); TF-IDF and offline opt-ins | Done |
| 3 | Packaged Domain/Function ModernBERT+logistic heads; cue is explicit fallback | Done |
| 4 | Wire fused taxonomy trees + exposed config; shared layers; docs | **This change** |
| 5 | App-family layer / with-app `0.03` dicts; diagnostic MiniLM flags; hardening | Not started |

Phase 4 default: `OrchardBuilder().build` (extras or fakes) produces named
`domain` + `function` trees from fused no-app matrices. Semantic stays
MiniLM/TF-IDF `0.66` / `0.34`. No tree named `fused` / `mixed`.
