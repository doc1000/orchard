# Orchard Architecture Decisions

## 1. Purpose

Orchard is a local-first Python library for converting a **fixed corpus of roughly 100–2,000 items** into persistent hierarchical trees that can be labeled, cut, pruned, queried, subset, and visualized.

The core separation is:

* **`OrchardBuilder` builds the persistent hierarchical schema.**
* **`Orchard` holds and manipulates the resulting trees.**
* Cuts and subtrees are cheap query-time views over canonical trees.
* AppWorld is a separate evaluation/demo consumer, not part of Orchard.

Primary scope: finite-corpus construction and manipulation.

---

## 2. Reference Models

Use these libraries as conceptual references rather than copying their implementation.

### BERTopic — top-level usability and modular pipeline

Orchard should feel similar to BERTopic from the outside:

```python
builder = OrchardBuilder()
orchard = builder.build(documents)
```

Simple defaults should produce something useful, while advanced users can inspect and replace internal components.

BERTopic is a useful model because it exposes a simple top-level object while supporting modular embeddings and other interchangeable stages.

### scikit-learn — component conventions

Use sklearn-style conventions where they fit naturally:

```python
taxonomy.fit(training_data)
taxonomy.transform(documents)
taxonomy.save(...)
```

Configuration should be inspectable and learned state separate from constructor configuration. Scikit-learn formalizes estimators and transformers around these conventions.

Do **not** force sklearn terminology where it is misleading. Orchard construction uses `build()`, not `fit_transform()`.

### SciPy hierarchy — canonical tree representation

The canonical binary tree remains SciPy-compatible linkage/Z-array based. SciPy linkage explicitly represents hierarchical/agglomerative clustering from observations or condensed distances, and its hierarchy module separately supports flattening/cutting those trees.

Orchard adds persistence, identity, labels, dynamic cutting, subtrees, and visualization around that representation.

---

## 3. Core Pipeline

### Construction

```text
raw records
→ canonical document records
→ optional taxonomy enrichment
→ embeddings / feature layers
→ similarity or distance matrices
→ fused profiles
→ canonical linkage trees
→ optional node labels
→ persisted Orchard
```

Construction is relatively expensive and produces durable artifacts.

Default named trees use **per-tree** fused profiles (independent MiniLM,
document TF-IDF, and taxonomy Jensen–Shannon layers). There is no orchard
tree named `fused` or `mixed`. Weights and `fusion_mode` (`variance_calibrated`
by default) are inspectable via `OrchardBuilder.get_params()` and overridable
through `profiles` / weight maps. Do not concatenate heterogeneous feature
vectors. See `docs/DECISIONS.md`.

### Manipulation

```text
canonical tree
→ score
→ cut / walk-cut
→ widen / deepen / prune
→ select subtree
→ visualize / export
```

Manipulation should normally be cheap, deterministic, and repeatable.

Manipulation must also work on externally supplied linkage matrices without requiring Orchard's construction pipeline.

---

## 4. Public Objects

### `OrchardBuilder`

Creates an Orchard from a finite corpus.

```python
orchard = OrchardBuilder().build(documents)
```

Responsibilities:

* normalize input;
* apply optional taxonomy enrichment;
* generate embeddings and similarity layers;
* fuse configured profiles;
* construct canonical binary trees;
* optionally label nodes;
* persist build artifacts.

### `Orchard`

Persistent collection of canonical trees and shared artifacts.

Contains references to:

* normalized corpus;
* trees;
* taxonomies;
* embeddings;
* matrices where retained;
* manifests and provenance.

### `Tree`

Canonical binary hierarchy.

Required identity/state:

* linkage/Z-array;
* stable canonical node IDs;
* ordered leaf IDs;
* memberships;
* linkage distances;
* node descriptions;
* optional named label sets.

The canonical tree is the source of truth.

### `ClusterView`

Cheap derived cut over canonical nodes.

Supports:

* target cluster count;
* top-level width;
* recursive target width;
* walk-cut;
* scoring;
* pruning;
* named cached views.

### `Subtree`

Reduced query-time version of a tree.

It retains canonical node identity and labels while using a local reduced linkage array.

It must itself support cutting, further subtree selection, and plotting.

### `Taxonomy` / `TaxonomyModel`

Input semantic classification structures.

Examples:

* domain;
* function;
* legal area;
* risk category.

Typical API:

```python
taxonomy = TaxonomyModel.from_labels(labels)
taxonomy.fit(training_data)
enriched = taxonomy.transform(documents)
```

Default pretrained domain and function taxonomies should ship with Orchard.

Taxonomy training is separate from tree construction.

---

## 5. Taxonomy vs Facet

Keep these concepts distinct.

**Taxonomy:** semantic structure supplied before linkage and used as an input signal.

**Facet:** an emergent category or navigation structure derived from the resulting trees.

Engineering code should generally use `taxonomy`.

Consumer-facing search/navigation systems may use `facet`.

---

## 6. Input Contract

All adapters normalize to a canonical record:

```text
item_id
text
title
metadata
source
```

Only `text` is required.

Defaults:

* generate `item_id` when absent;
* allow empty title;
* empty metadata mapping;
* optional source.

Supported initial adapters:

* list of strings;
* list of mappings;
* JSON / JSONL;
* CSV;
* directory of documents.

No pandas dependency is required.

For this release, one normalized record equals one tree leaf.

---

## 7. Persistence

Use a **versioned artifact directory + manifest**, not one large JSON object.

### Durable

* normalized corpus;
* canonical linkage trees;
* node registry;
* taxonomy models/definitions;
* embeddings;
* named label sets;
* build configuration;
* provenance/checksums.

### Optional / policy-based

* feature layers;
* fused similarity/distance matrices;
* vector indexes.

Dense matrix persistence should be configurable:

```text
always
never
below_size_limit
compressed
```

NPZ is sufficient initially.

### Cached / derived

* named cuts;
* scoring output;
* Plotly structures.

### Ephemeral

* temporary cuts;
* working subtrees;
* transient figures.

---

## 8. Node Labeling

Tree structure must work without LLM labeling.

Labeling happens after linkage construction.

Support separate named label sets:

```python
tree.labels["intrinsic_v1"]
tree.labels["contrastive_v1"]
```

Initial modes:

* none;
* intrinsic;
* contrastive;
* user-supplied.

Relabeling must never mutate linkage, memberships, or canonical node IDs.

---

## 9. Tree Manipulation

Preserve existing functionality for:

* dynamic cuts;
* target cluster counts;
* width-constrained cuts;
* walk-cut;
* cut scoring;
* pruning;
* subtree extraction;
* Plotly hierarchy generation.

Cuts and subtrees should reference canonical nodes rather than reconstructing semantic identity.

A subtree uses:

* local linkage indexes for computation;
* canonical IDs for identity/provenance.

---

## 10. External Linkage Support

A user should be able to begin with:

```python
Tree.from_linkage(
    linkage=z,
    leaf_ids=ids,
    leaf_payloads=records,
)
```

Optional inputs may include:

* node registry;
* embeddings;
* distance matrix.

The resulting tree must support Orchard's manipulation layer:

* labeling;
* cutting;
* subtrees;
* plotting;
* persistence.

This keeps tree manipulation independent from OrchardBuilder.

---

## 11. Explicitly Out of Scope

Do not implement in this refactor:

* incremental insertion;
* deletion or mutation of existing corpus records;
* online tree maintenance;
* SQL/Postgres persistence;
* distributed processing;
* hosted service;
* multi-user support;
* cross-tree synchronized navigation;
* automatic best-tree selection;
* very large corpus optimization;
* migration of historical planning material.

Future `transform(new_documents)` may implement projection and Z-array grafting, but should remain absent or explicitly stubbed for now.

---

## 12. Refactor Rules

The extraction should:

* preserve working algorithms before redesigning them;
* remove AppWorld-specific logic from Orchard;
* remove public phase-number terminology;
* promote authoritative notebook/adaptor implementations into normal modules;
* remove obsolete experimental tests and planning artifacts;
* retain compact contract/invariant tests defined in `RELEASE_GATE.md`;
* keep defaults inspectable;
* avoid unnecessary changes to the modeling stack;
* favor a small public API over minimizing total source-code size.

The README should contain one complete minimal workflow and one advanced workflow. Both should run as tests.

---

## Working Definition

> **Orchard builds persistent hierarchical schemas over finite corpora and exposes them as dynamically cuttable, prunable, labelable, queryable, and visualizable trees.**
