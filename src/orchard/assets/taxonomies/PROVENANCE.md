# Taxonomy asset provenance

Default Domain and Function definitions are genericized from the human-approved
`taxonomy_v0` label inventories used in the AppWorld tool-tree demo
(`tool-tree-demo/artifacts/appworld/phase2b/taxonomies/taxonomy_v0/`).

Adaptations for Orchard:

- AppWorld corpus hashes, tool examples, and app-name overrides removed
- Cue sets retained as offline transform signals (no ModernBERT / joblib heads)
- Tree names are `domain` and `function` (demo used `functional` for the latter)

Packaged ModernBERT+logistic student heads live in `heads/` (see
`heads/PROVENANCE.md`). Cue sets remain the offline fallback transform.
When `orchard[taxonomy-ml]` is installed, default Domain/Function
`transform()` uses those heads (`modernbert_logistic`), not cues.

Replace these JSON files or call `TaxonomyModel.load(...)` / `.fit(...)` /
`.load_head(...)` to swap definitions or heads without changing Orchard
linkage code.
