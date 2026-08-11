# RELEASE GATE

Gates apply to the standalone Orchard repository. Use generic fixtures only.

## Cross-cutting (every phase)

- [ ] `CODE_MANIFEST.md` is updated; shipped items have no `[proposed]` tag.

## Product gates

- [ ] Clean install succeeds in an empty environment (no external eval suite required).
- [ ] Generic fixtures load and satisfy identity contracts.
- [ ] Tiny corpus builds with no API keys and no taxonomies → Orchard with a `semantic` tree.
- [ ] Default Domain + Function taxonomy build succeeds → Orchard with both named trees.
- [ ] `Orchard` is multi-tree: `orchard.trees` / `orchard.tree(name)` work after save/load.
- [ ] Save/load preserves shared document IDs and per-tree canonical node identity.
- [ ] Every cut partitions the active leaf set exactly.
- [ ] Subtrees retain canonical identity and remain recursively usable.
- [ ] External linkage trees support cutting, plotting, persistence, and label import.
- [ ] Switching or replacing named label sets never mutates linkage.
- [ ] Visualization payloads validate against their current source tree and support multiple trees.
- [ ] Incremental corpus mutation is explicitly unsupported.
- [ ] README minimal example runs as an automated test.
- [ ] Local documentation site opens without a build server and covers the public API areas.
