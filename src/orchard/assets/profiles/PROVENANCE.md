# Packaged similarity profiles

These weight dicts already sum to 1. Orchard never silently renormalizes.
Missing layers are **absent**: ship the matching dict, do not fold at runtime.

Layer name `function_raw_js` (tree id `function`) is AppWorld
`functional_raw_js`. There is no orchard tree named `fused` or `mixed`.
Layer name `app_exact_match` is AppWorld exact `app_name` equality.

| Tree | When | Weights |
|---|---|---|
| `semantic` | always (family on or off) | `description_minilm_centered_cosine=0.66`, `tfidf_cosine=0.34` |
| `domain` | both taxonomies, no app | `domain_raw_js=0.48`, MiniLM `0.23`, TF-IDF `0.14`, `function_raw_js=0.15` |
| `function` | both taxonomies, no app | `function_raw_js=0.48`, MiniLM `0.25`, TF-IDF `0.15`, `domain_raw_js=0.12` |
| `domain` | only domain, no app | `domain_raw_js=0.63`, MiniLM `0.23`, TF-IDF `0.14` |
| `function` | only function, no app | `function_raw_js=0.60`, MiniLM `0.25`, TF-IDF `0.15` |
| `domain` | both taxonomies + app | `app_exact_match=0.03`, MiniLM `0.23`, `domain_raw_js=0.45`, `function_raw_js=0.15`, TF-IDF `0.14` |
| `function` | both taxonomies + app | `app_exact_match=0.03`, MiniLM `0.25`, `domain_raw_js=0.12`, `function_raw_js=0.45`, TF-IDF `0.15` |
| `domain` | only domain + app | `domain_raw_js=0.60`, MiniLM `0.23`, TF-IDF `0.14`, `app_exact_match=0.03` |
| `function` | only function + app | `function_raw_js=0.57`, MiniLM `0.25`, TF-IDF `0.15`, `app_exact_match=0.03` |

No-app dicts fold the 0.03 AppWorld `app_exact_match` mass into the dominant
taxonomy JS. With-app dicts are the AppWorld G3 numbers
(`keyword_tfidf_cosine` → `tfidf_cosine`). They are selected only when the
family layer is actually enabled (complete metadata **and** caller opt-in).
There is no packaged semantic-with-app default.

Default fusion mode is `variance_calibrated` (n≥3; a selected layer with
off-diagonal variance ≤ `1e-12` aborts). `raw_convex` remains available.
