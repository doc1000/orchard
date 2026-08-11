# Orchard

Local-first Python library for building persistent, cuttable hierarchical trees over a fixed document corpus.

## Install

```bash
uv pip install -e ".[dev]"
```

## Minimal example

```python
from orchard import OrchardBuilder

orchard = OrchardBuilder().build(
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

Semantic-only (no taxonomies):

```python
orchard = OrchardBuilder(taxonomies=[]).build(documents)
semantic = orchard.tree("semantic")
```

## Docs

Open [`docs/site/index.html`](docs/site/index.html) in a browser (no build server). See also `docs/ORCHARD_ARCHITECTURE.md` and `CODE_MANIFEST.md`.
