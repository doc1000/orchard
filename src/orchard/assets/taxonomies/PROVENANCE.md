# Taxonomy asset provenance

Default Domain and Function definitions are genericized from the human-approved
`taxonomy_v0` label inventories used in the AppWorld tool-tree demo
(`tool-tree-demo/artifacts/appworld/phase2b/taxonomies/taxonomy_v0/`).

Adaptations for Orchard:

- AppWorld corpus hashes, tool examples, and app-name overrides removed
- Cue sets retained as offline transform signals (no ModernBERT / joblib heads)
- Tree names are `domain` and `function` (demo used `functional` for the latter)

Replace these JSON files or call `TaxonomyModel.load(...)` / `.fit(...)` to
swap definitions without changing Orchard linkage code.
