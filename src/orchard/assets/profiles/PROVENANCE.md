# Packaged no-app similarity profiles

These weight dicts already sum to 1. Orchard never silently renormalizes.
Missing layers are **absent**: ship the matching dict, do not fold at runtime.

Layer name `function_raw_js` (tree id `function`) is AppWorld
`functional_raw_js`. There is no orchard tree named `fused` or `mixed`.

| Tree | When | Weights |
|---|---|---|
| `semantic` | always, no app | `description_minilm_centered_cosine=0.66`, `tfidf_cosine=0.34` |
| `domain` | both taxonomies, no app | `domain_raw_js=0.48`, MiniLM `0.23`, TF-IDF `0.14`, `function_raw_js=0.15` |
| `function` | both taxonomies, no app | `function_raw_js=0.48`, MiniLM `0.25`, TF-IDF `0.15`, `domain_raw_js=0.12` |
| `domain` | only domain, no app | `domain_raw_js=0.63`, MiniLM `0.23`, TF-IDF `0.14` |
| `function` | only function, no app | `function_raw_js=0.60`, MiniLM `0.25`, TF-IDF `0.15` |

The 0.03 AppWorld `app_exact_match` mass is folded into the dominant taxonomy
JS. With-app dicts are Phase 5 only.

Default fusion mode is `variance_calibrated` (n≥3; a selected layer with
off-diagonal variance ≤ `1e-12` aborts). `raw_convex` remains available.
`keyword_tfidf_cosine` is renamed to document-text `tfidf_cosine`.
