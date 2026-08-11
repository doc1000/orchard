# Orchard

Local-first Python library for building persistent, cuttable hierarchical trees over a fixed document corpus.

Install (editable):

```bash
uv pip install -e ".[dev]"
```

Phase 0 provides document and identity contracts plus generic fixtures. Construction (`OrchardBuilder.build`), multi-tree `Orchard`, and the docs site arrive in later phases — see `docs/ORCHARD_EXTRACTION_PLAN.md`.
