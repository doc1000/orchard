# Orchard Extraction Plan

> **Historical.** This document records how Orchard was first carved out of
> the demo. Do **not** treat its non-goals as current product law. Per-tree
> fused profiles are in scope (`docs/DECISIONS.md`,
> `docs/FUSED_SIMILARITY_PLAN.md`). There is still no single orchard tree
> named `fused` / `mixed`.

Standalone extraction of working Orchard logic from `../tool-tree-demo` (reference only; do not refactor it).

**Architecture source:** `docs/ORCHARD_ARCHITECTURE.md`  
**Ignore:** `docs/archive/ORCHARD_REFACTOR_DECISIONS.md` (archived; do not drive work)  
**Done when:** `docs/RELEASE_GATE.md` product gates pass.

## Target architecture (summary)

- `Orchard` is multi-tree by default over one shared document corpus.
- Configured taxonomies → **one named tree per taxonomy** (per-tree fusion is in scope; still not a mixed orchard tree named `fused`).
- No taxonomies → single **semantic** tree from semantic similarity.
- Built-in Domain and Function taxonomy artifacts are first-class examples (no AppWorld dependency).
- Public construction verb is `build`.
- Linkage is immutable w.r.t. labels; named label sets are presentation overlays.
- v1 labels: intrinsic, imported/user-provided, named sets. Contrastive = future work.
- Policy: extract → clean boundaries → regression test → expose. No algorithm redesign.

### Intended workflows

```python
# Taxonomies → one tree each
builder = OrchardBuilder(taxonomies=[DomainTaxonomy.load_default(), FunctionTaxonomy.load_default()])
orchard = builder.build(documents)
orchard.tree("domain"); orchard.tree("function")

# No taxonomies → semantic tree
orchard = OrchardBuilder(taxonomies=[]).build(documents)
orchard.tree("semantic")

# External linkage
tree = Tree.from_linkage(Z, item_ids=item_ids)
orchard = Orchard.from_trees(documents=documents, trees={"custom": tree})
```

Signatures may refine; keep this shape.

### Non-goals

AppWorld / retrieval / BigTool; Postgres Orchard; online mutation / grafting; auto best-tree; cross-tree navigation; ~~fused default trees~~ *(reversed for **per-tree** fusion; still no mixed orchard tree)*; contrastive label generation; distributed/hosted features; vector DB productization; major numerical redesign; heavy validation frameworks.

---

## Phase 0 — Repository contracts + code manifest

**Goal.** Minimal stable foundation; no empty public facades.

**Source functionality to extract.** Identity/schema ideas from reference tree JSON shapes and document fields (`item_id`/`text`/…); nothing algorithmic yet.

**Work.**
- Package skeleton + dependency baseline.
- `Document`; shared item IDs; tree IDs; canonical node IDs; schema/artifact version.
- Local linkage index ↔ canonical node ID contract.
- Exception types only as needed (incl. unsupported corpus mutation).
- Small generic reference fixtures (documents + sample linkage).
- Create `CODE_MANIFEST.md` (proposed layout allowed with `[proposed]`).

**Outputs.** Installable package; contracts + fixtures; `CODE_MANIFEST.md`.

**Release gate.**
- Clean install succeeds.
- Fixtures load; identity/uniqueness contracts tested.
- `CODE_MANIFEST.md` reflects current + proposed structure.

---

## Phase 1 — Core multi-tree Orchard + tree manipulation

**Goal.** Useful Orchard when the user supplies linkage; multi-tree viz/persist.

**Source functionality to extract.**
- Canonical tree build/validators: `tool_tree_demo/phase4a.py`
- Dynamic cut + Steiner subtree: `adapters/dynamic_tree_cutter.py`
- Packing (if still useful): `adapters/linkage_cluster_packer.py`
- Plotly payloads: `adapters/plotly_nested_structure.py`
- Walk helpers: `notebooks/streamlined.ipynb` (`walk_cut`, `walk_cut_json`)

**Work.**
- `Orchard` owning shared documents + named `Tree`s; `Orchard.from_trees`, save/load.
- `Tree.from_linkage`; canonical construction; membership/node registry.
- Cuts, subtree/local-Z, walk/view helpers, packing if retained.
- Named label **storage** (for plot/display); no automated labeling yet.
- Visualization payloads that enumerate/select among multiple trees.
- Compact invariant tests on generic fixtures (partition, ID maps, linkage shape).

**Outputs.** Multi-tree Orchard usable from external Z; persisted artifact directory.

**Release gate.**
- Save/load preserves shared docs + per-tree canonical IDs for ≥2 trees.
- Cuts partition active leaves; subtrees recursive + identity-preserving.
- Viz payloads validate vs source tree; multi-tree representation works.
- External linkage path: cut / plot / persist / import labels.
- `CODE_MANIFEST.md` updated (promoted modules no longer `[proposed]`).

---

## Phase 2 — Build trees from documents

**Goal.** Public `OrchardBuilder.build(documents)` with taxonomy-vs-semantic branching.

**Source functionality to extract.**
- Representations / similarity / fuse pieces as needed per **tree**, not as one mixed orchard tree: `phase3.py`, `phase3b.py` (numeric cores only).
- Embedding bridges: `adapters/phase3_embedding_bridge.py`, `sentence_embedding_bridge.py` (clean into backends).
- Linkage path already in Phase 1.

**Work.**
- `build`: taxonomies configured → one tree per taxonomy name; `taxonomies=[]` → `semantic` tree only.
- No default Domain+Function+semantic fused tree.
- Wire inspectable builder config; preserve working numeric behavior.
- Defaults that make Domain/Function easy once Phase 3 artifacts land (hooks OK; full artifact packaging in Phase 3).
- Regression tests: tiny corpus no-taxonomy → `semantic`; stub/fixture taxonomies → one tree each.

**Outputs.** End-to-end build without AppWorld; semantic corner case tested.

**Release gate.**
- Tiny corpus, no API keys, no taxonomies → `orchard.tree("semantic")`.
- Multi-taxonomy build produces distinct named trees (fixture taxonomies acceptable until Phase 3 defaults).
- No mixed/fused default tree in API or docs.
- `CODE_MANIFEST.md` updated.

---

## Phase 3 — Taxonomy artifacts + labeling

**Goal.** Reusable taxonomies + label overlays; Domain/Function usable out of the box.

**Source functionality to extract.**
- Taxonomy definitions + trained artifacts: `taxonomy_v0.py`, classification path in `classification.py`, classifier bridge (strip AppWorld overrides).
- Intrinsic labeling: `phase4b1.py` (+ optional OpenAI runner as a backend, not a hard dependency).
- **Do not** extract contrastive generation (`phase4_contrast.py`) into v1 API.

**Work.**
- First-class taxonomy load/transform path; user-defined taxonomies where already practical.
- Choose packaging: bundle Domain/Function trained artifacts **or** standalone loadable assets—inspectable, replaceable, no AppWorld assumptions.
- Document artifact provenance (from prior demo/enrichment work → `orchard` assets path).
- Intrinsic labeling; import user labels; named sets; `use_labels` without touching linkage.
- Prove label swap does not rebuild or invalidate linkage/caches keyed on structure.

**Outputs.** Default Domain + Function examples; intrinsic + imported labels.

**Release gate.**
- Default Domain + Function build → `domain` and `function` trees.
- Artifacts inspectable/replaceable; no AppWorld dependency or overrides.
- Label set switch/replace never mutates linkage.
- Intrinsic and imported labels work; contrastive absent from public API.
- `CODE_MANIFEST.md` updated.

---

## Phase 4 — Adapters + productization

**Goal.** Convenience I/O, short README, local docs site, ship-ready package.

**Source functionality to extract.** None required beyond normalizing patterns already used for JSON/JSONL in the reference repo.

**Work.**
- Lightweight adapters → `Document`: directory, JSON/JSONL, CSV, list/dict records.
- Concise README (what / install / minimal example / link to docs).
- Local multi-page docs site (static HTML/CSS/JS, VS Code-style sidebar): Getting Started, Architecture, Documents, Taxonomies, Building Trees, Multiple Trees, Cuts & Views, Labels, Persistence, Visualization, Adapters, Extending Orchard.
- Public API cleanup; optional dependency groups; lint/type/test config as useful.
- Clean-env install + README example test; optional cheap smoke timings.
- Generic examples only.

**Outputs.** Usable standalone product; `CODE_MANIFEST.md` as final map.

**Release gate.**
- Adapters normalize into `Document` and feed `build`.
- README minimal example automated.
- Docs site opens locally without a build server; sidebar covers API areas.
- Clean-environment install/run passes.
- All product gates in `RELEASE_GATE.md` checked.
- `CODE_MANIFEST.md` final (no stale `[proposed]` for shipped surface).

---

## Appendix A — Source map (compact)

| Area | Reference path | Promote? |
|------|----------------|----------|
| Canonical tree / validators | `src/tool_tree_demo/phase4a.py` | Yes |
| Dynamic cut / subtree Z | `adapters/dynamic_tree_cutter.py` | Yes |
| Packing | `adapters/linkage_cluster_packer.py` | Yes if still used |
| Plotly rows | `adapters/plotly_nested_structure.py` | Yes |
| Walk views | `notebooks/streamlined.ipynb` | Yes |
| Similarity / layers | `phase3.py`, `phase3b.py` | Numeric cores; per-tree use |
| Embeddings | `adapters/*embedding*bridge.py` | Clean backends |
| Taxonomy defs / classify | `taxonomy_v0.py`, `classification.py`, classifier bridge | Yes; strip AppWorld |
| Intrinsic labels | `phase4b1.py` | Yes |
| Contrastive | `phase4_contrast.py` | Future only |
| AppWorld load / retrieval / BigTool / Phase 5C UI | `loader.py`, `retrieval/**`, related adapters, `web/` | No |
| graph-engine DB orchard | sibling repo | No |

## Appendix B — Manifest & docs rules

- `CODE_MANIFEST.md` at repo root: indented map; one-line descriptions; list important public symbols under each code file; `[proposed]` until implemented; update every phase.
- README stays short; architecture and API live in `docs/` site pages.
- Do not treat `TREE_PIPELINE_PROCESS.html` as product docs; it is reference process history from the demo.
