# RELEASE GATE

Gates apply to the standalone Orchard repository. Use generic fixtures only.

## Cross-cutting (every phase)

- [x] `CODE_MANIFEST.md` is updated; shipped items have no `[proposed]` tag.

## Product gates

- [x] Clean install succeeds in an empty environment (no external eval suite required).
- [x] Generic fixtures load and satisfy identity contracts.
- [x] Tiny corpus builds with no API keys and no taxonomies → Orchard with a `semantic` tree.
- [x] Default Domain + Function taxonomy build succeeds → Orchard with both named trees.
- [x] `Orchard` is multi-tree: `orchard.trees` / `orchard.tree(name)` work after save/load.
- [x] Save/load preserves shared document IDs and per-tree canonical node identity.
- [x] Every cut partitions the active leaf set exactly.
- [x] Subtrees retain canonical identity and remain recursively usable.
- [x] External linkage trees support cutting, plotting, persistence, and label import.
- [x] Switching or replacing named label sets never mutates linkage.
- [x] Visualization payloads validate against their current source tree and support multiple trees.
- [x] Incremental corpus mutation is explicitly unsupported.
- [x] README minimal example runs as an automated test.
- [x] Local documentation site opens without a build server and covers the public API areas.
