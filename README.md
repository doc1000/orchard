# Orchard

Local-first Python library for building persistent, cuttable hierarchical trees over a fixed document corpus.

## Install

```bash
uv pip install -e ".[dev]"
```

Neural defaults (MiniLM description layer + packaged Domain/Function heads):

```bash
uv pip install -e ".[dev,ml]"
```

## Minimal example (offline)

Clean environments without `orchard[embeddings]` / `orchard[taxonomy-ml]` must opt in. This path is single-layer TF-IDF / cue JS.

```python
from orchard import OrchardBuilder

orchard = OrchardBuilder(allow_offline_fallback=True).build(
    [
        "Schedule a calendar reminder for tomorrow morning.",
        "Send an email summary of the weekly status notes.",
        "Create a task to review the draft proposal.",
        "Find documents mentioning quarterly budget planning.",
    ]
)
assert orchard.tree_ids == ("domain", "function")
domain = orchard.tree("domain")
```

Semantic-only (explicit TF-IDF):

```python
from orchard.backends.tfidf import TfidfEmbeddingBackend

orchard = OrchardBuilder(
    taxonomies=[],
    embedding_backend=TfidfEmbeddingBackend(),
).build(documents)
semantic = orchard.tree("semantic")
```

## Advanced (neural extras)

With `orchard[ml]` (or injected MiniLM + taxonomy encoder), `OrchardBuilder().build`
produces named `domain` + `function` trees from **fused** no-app profiles
(`variance_calibrated`). Semantic is MiniLM `0.66` / TF-IDF `0.34`. Inspect
weights with `builder.get_params()`. There is no tree named `fused` / `mixed`.

```python
from orchard import OrchardBuilder

builder = OrchardBuilder()
orchard = builder.build(documents)
params = builder.get_params()
assert params["fusion_mode"] == "variance_calibrated"
assert orchard.tree_ids == ("domain", "function")
```

Variance-calibrated fusion needs n≥3 documents.

## Docs

Open [`docs/site/index.html`](docs/site/index.html) in a browser (no build server). See also `docs/ORCHARD_ARCHITECTURE.md`, `docs/DECISIONS.md`, and `CODE_MANIFEST.md`.
