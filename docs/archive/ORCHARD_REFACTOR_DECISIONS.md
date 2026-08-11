# Orchard Separation and Refactor Decisions

## Purpose

Orchard will be separated from the AppWorld evaluation suite and developed as a standalone, notebook-friendly Python library.

Its primary purpose is to take a **fixed, finite corpus** and turn it into one or more persistent hierarchical trees that can be labeled, cut, pruned, queried, visualized, and reduced into temporary subtrees.

The core metaphor is:

* **OrchardBuilder cultivates the schema.**
* **Orchard holds the cultivated trees.**
* **Cuts, pruned trees, and subtrees are query views—the fruit produced from the orchard.**

The AppWorld repository will remain separate as an evaluation, benchmarking, and demonstration environment.

---

## Primary Product

The current product is a **finite-corpus builder for approximately 100–2,000 items**.

Typical use cases:

### Exploratory

* Supply a list of tool descriptions, documents, or records.
* Use default enrichment and similarity settings.
* Build several trees.
* Inspect, cut, label, and visualize the resulting structure.

### Structured

* Supply records following a defined ingestion contract.
* Load or train custom taxonomies.
* Configure similarity layers and tree profiles.
* Persist the resulting trees for repeated querying and downstream applications.

New-document insertion, deletion, modification, and online tree maintenance are future capabilities.

---

## Architectural Boundary

The system is divided into two major processes.

### 1. Construction

Construction creates the durable schema:

```text
raw records
→ normalized documents
→ optional taxonomy enrichment
→ embeddings and feature layers
→ similarity or distance matrices
→ fused profiles
→ canonical binary trees
→ optional node labeling
→ persisted Orchard
```

Construction may be relatively expensive. It can involve model inference, dense pairwise calculations, hierarchical linkage, and LLM labeling.

### 2. Manipulation

Manipulation queries or reshapes an existing schema:

```text
canonical tree
→ score
→ cut
→ walk cut
→ widen or deepen
→ prune
→ select subtree
→ produce Plotly structure
```

These operations should normally be cheap, deterministic, and repeatable.

A major design requirement is that manipulation must also work when the user supplies a linkage matrix and node payload created outside Orchard.

This separation is fundamental:

> Construction builds a populated hierarchical schema. Manipulation queries and reshapes that schema.

---

## Public Vocabulary

### Taxonomy

A **taxonomy** is an input semantic classification structure that contributes information before or during linkage construction.

Examples:

* domain;
* function;
* legal area;
* document type;
* risk category.

A taxonomy may be pretrained, user-defined, or learned from labeled data.

The engineering API should use terms such as:

```python
Taxonomy
TaxonomyModel
TaxonomyResult
```

### Facet

A **facet** is an emergent or consumer-facing category derived from the resulting tree structure.

Taxonomies and facets are therefore intentionally different:

> Taxonomies add semantic structure to the inputs. Facets emerge from the resulting hierarchy and are used to navigate or query it.

The front end may use the word `facet` even when the underlying construction process used one or more taxonomies.

---

## Proposed Public Objects

### `OrchardBuilder`

Builds persistent Orchard artifacts from a finite corpus.

Responsibilities:

* normalize supported inputs;
* invoke optional enrichment;
* generate embeddings and feature layers;
* generate and fuse similarity profiles;
* build canonical linkage trees;
* optionally invoke node labeling;
* persist the resulting Orchard.

The canonical public verb is:

```python
orchard = builder.build(documents)
```

`build()` means:

> Take the complete current corpus payload and construct new canonical trees.

It does not mean incremental insertion into an existing tree.

`fit_transform()` should not be the primary Orchard-level API. True `fit` and `transform` semantics belong more naturally to enrichment and taxonomy components.

### `Orchard`

Represents the durable, populated hierarchical schema.

It contains references to:

* normalized corpus registry;
* canonical trees;
* node records and descriptions;
* taxonomy definitions and models;
* embeddings;
* optional similarity or distance matrices;
* manifests and provenance.

It exposes tree selection, cutting, pruning, subtree extraction, labeling, plotting, and artifact export.

### `Tree`

Represents one canonical binary hierarchy.

Minimum durable state:

* SciPy-compatible linkage or Z-array;
* stable leaf and internal node IDs;
* node membership references;
* linkage distances;
* node descriptions;
* label sets, when available;
* provenance;
* construction configuration.

The canonical tree is the source of truth. Cuts, subtrees, and visual structures are derived from it.

### `ClusterView`

Represents a temporary or cached cut of a tree.

Examples:

```text
domain_default
functional_optimized
wide_navigation
target_25
```

A view stores its configuration and references canonical node IDs. It should not duplicate the canonical tree unnecessarily or invent new memberships.

### `Subtree`

Represents a reduced, query-time version of a canonical tree.

A subtree is a “mini-me” of the parent tree:

* reduced Z-array;
* inherited canonical node IDs;
* inherited linkage distances;
* inherited labels and descriptions;
* references to the same source-of-truth node registry.

A subtree must support the same core manipulation operations as a primary tree:

* cutting;
* scoring;
* selecting further subtrees;
* plotting.

Subtrees are normally ephemeral, but may be saved when they represent a useful result of iterative exploration.

### `TaxonomyModel`

Handles taxonomy training and inference.

Expected pathways:

```python
TaxonomyModel.load(...)
TaxonomyModel.from_labels(...)
TaxonomyModel.fit(labeled_documents)
TaxonomyModel.transform(documents)
TaxonomyModel.save(...)
```

Taxonomy training is independent from Orchard construction.

---

## Enrichment Decisions

Enrichment should have its own pathways and should not be fused conceptually with tree construction.

Orchard construction must support:

1. no taxonomy enrichment;
2. bundled pretrained taxonomies;
3. user-supplied pretrained taxonomies;
4. custom taxonomies trained separately.

The default distribution should include pretrained:

* domain;
* function.

`OrchardBuilder.build()` may use these defaults unless disabled by configuration.

Taxonomy work should follow conventional estimator semantics where useful:

```python
taxonomy.fit(training_data)
taxonomy.transform(documents)
```

The Orchard itself uses:

```python
builder.build(documents)
```

Tree construction must work without LLM-generated node labels.

Intrinsic and contrastive node labeling are separate post-tree operations that can be:

* run;
* rerun;
* compared;
* cached;
* omitted.

This allows structural quality and label quality to be explored independently.

---

## Label Sets

Labels should be treated as revisions or named overlays rather than destructive mutations of the tree.

Conceptually:

```python
tree.labels["intrinsic_v1"]
tree.labels["contrastive_v2"]
tree.use_labels("contrastive_v2")
```

Rerunning labeling should not alter:

* linkage;
* canonical node IDs;
* memberships;
* distances.

A tree may therefore accumulate multiple named label sets.

Supported labeling modes may include:

* no labels;
* intrinsic labels;
* contrastive labels;
* user-supplied labels;
* future custom labeling strategies.

---

## Input Contract

Orchard will normalize supported inputs into one canonical document record.

Conceptually:

```text
item_id
text
title
metadata
source
```

Requirements:

* `text` is the only required semantic field;
* `item_id` may be supplied or generated;
* `title` may be supplied, inferred, or empty;
* `metadata` defaults to an empty mapping;
* `source` is optional;
* leaf node IDs may default to item IDs.

Supported adapters may include:

* list of strings;
* list of mappings;
* JSON or JSONL;
* CSV;
* directory of documents;
* externally constructed records.

The internal contract must not require pandas.

The current scope treats each input document or record as one leaf item.

Existing chunking capability may remain available, but chunking policy should not complicate or block the first standalone release.

The normalized corpus registry should be a required durable artifact because it provides the stable identity and payload needed for rebuilding, tracing, labeling, validation, and future corpus mutation.

---

## Persistence Model

Persistence is central to Orchard.

The preferred model is a **versioned artifact directory with a manifest**, not one giant JSON file.

### Durable Artifacts

Expensive or authoritative artifacts should persist by default or through an explicit build destination.

These include:

* normalized corpus registry;
* taxonomy definitions;
* trained taxonomy model weights;
* embeddings;
* canonical linkage trees;
* node registries;
* node label sets;
* construction manifests;
* provenance and checksums.

For each tree, required persistent state should include:

* linkage or Z-array;
* node registry;
* leaf membership references;
* node descriptions;
* label sets, when generated;
* construction configuration;
* provenance;
* checksums.

### Important Secondary Artifacts

These are valuable but not always required:

* per-layer feature outputs;
* fused similarity or distance matrices;
* cached named cuts;
* vector-search indexes.

Embeddings should normally persist because they may support:

* hybrid search;
* tree traversal;
* future projection;
* future grafting;
* candidate-node retrieval.

Dense matrices may persist conditionally according to corpus size and configuration.

A simple policy should be supported:

```text
always
never
below_size_limit
compressed
```

NPZ is acceptable for the first implementation.

Parquet may be useful for tabular corpus and metadata records, but it should not be forced onto dense numerical matrices.

A future vector index may use FAISS or another lightweight local structure without requiring a relational database.

---

## Matrix Scale

The current target is approximately **100–2,000 items** on a capable laptop CPU.

At this scale, dense matrices remain practical, but multiple feature layers and multiple fused profiles can increase storage quickly.

The design should therefore:

* keep matrix access available;
* avoid unnecessary matrix duplication;
* allow compact persistence;
* allow matrices to be omitted above a configurable size threshold;
* preserve enough linkage and cophenetic information for downstream manipulation when full matrices are unavailable.

A tree should reference the fused matrix and shared feature layers that produced it rather than duplicating large arrays unnecessarily.

---

## Artifact Lifecycle

Artifacts fall into three classes.

### Durable

Expensive or authoritative:

* normalized corpus registry;
* trained taxonomy models;
* embeddings;
* canonical linkage trees;
* node registries;
* node label sets;
* manifests.

### Reproducible Cache

Cheap but repeatedly useful:

* named cuts;
* Plotly row structures;
* scoring outputs;
* derived navigation views.

These may be cached with their configuration and invalidated when the parent tree or relevant label set changes.

### Ephemeral

Query-time working artifacts:

* temporary cuts;
* intermediate subtrees;
* pruned query views;
* transient Plotly figures.

These should remain in memory unless explicitly saved.

---

## Cluster Cutting and Tree Manipulation

The manipulation API must preserve the existing functional behavior, including:

* scoring candidate cuts;
* target cluster count;
* top-level width;
* recursive target width;
* dynamic cutting;
* walk-cut behavior;
* relabeling choices;
* pruning;
* reconstruction of presentation hierarchies;
* Plotly-compatible nested structures.

Cuts are primarily views over canonical nodes.

A named view should record:

* parent tree ID;
* cut strategy;
* parameters;
* scoring configuration;
* active label set;
* tree version;
* resulting canonical node IDs.

Named cached views may include examples such as:

```text
domain_default
functional_optimized
navigation_wide
```

The configuration is important to persist. The derived artifact itself should remain cheap to regenerate.

---

## Subtree Semantics

A subtree represents a reduced query artifact over an existing canonical tree.

Its purpose is to support iterative tree use:

```text
tree
→ select subset
→ construct subtree
→ cut subtree
→ select subset
→ construct smaller subtree
→ continue
```

The subtree should inherit the expensive work already performed on the parent:

* taxonomy-related structure;
* linkage information;
* node identity;
* distances;
* node labels.

The subtree should not require relabeling or rebuilding similarity structure unless explicitly requested.

A reduced linkage matrix uses local linkage indexing, while Orchard also requires stable canonical identity.

A subtree therefore needs both:

* local Z-array indexing for computation;
* canonical node IDs for identity, labels, and provenance.

This mapping must be explicit and must not be inferred solely from array position.

---

## External Tree Support

Orchard should allow a user to begin from an externally generated linkage tree.

At minimum, this pathway should accept:

* linkage matrix;
* ordered leaf IDs;
* optional leaf payloads;
* optional node registry;
* optional embeddings;
* optional distance matrix.

The user should then be able to use Orchard for:

* node labeling;
* cluster cutting;
* subtree extraction;
* plotting;
* persistence;
* export.

This is a first-class capability because it separates Orchard’s schema-manipulation tools from Orchard’s specific construction pipeline.

A user should not need `OrchardBuilder` in order to use Orchard manipulation tools on a valid external linkage tree.

---

## Simplified Usage Target

The README should include a complete default example that is also executable as an automated test.

```python
from orchard import OrchardBuilder

builder = OrchardBuilder()

orchard = builder.build(
    [
        "Create and send an invoice",
        "Search saved email messages",
        "Transfer money to a contact",
    ]
)

tree = orchard.tree("mixed")
view = tree.cut(target_clusters=10)
view.plot()
```

The default builder should be transparent and inspectable:

```python
builder.config
builder.taxonomies
builder.profiles
```

A user should be able to determine:

* which taxonomies are active;
* which embedding model is used;
* which similarity profiles will be built;
* which labeling strategy is configured.

Defaults should be convenient, but not magical.

A custom taxonomy workflow may look like:

```python
from orchard import OrchardBuilder, TaxonomyModel

legal = TaxonomyModel.from_labels(
    ["contracts", "litigation", "regulation", "corporate"]
)

legal.fit(labeled_documents)

builder = OrchardBuilder(
    taxonomies={
        "domain": "default",
        "function": "default",
        "legal": legal,
    },
    node_labeling=None,
)

orchard = builder.build(documents)
orchard.save("./artifacts/my_orchard")
```

Loading and manipulating an existing artifact should be equally simple:

```python
from orchard import Orchard

orchard = Orchard.load("./artifacts/my_orchard")

tree = orchard.tree("domain")
subtree = tree.subtree(item_ids=selected_ids)
view = subtree.walk_cut(target_width=8)
fig = view.plot()
```

---

## Remaining Decisions

The following details remain intentionally unresolved and should be handled during implementation without expanding scope unnecessarily.

### Matrix Persistence Threshold

A default cutoff for dense-matrix persistence has not been selected.

The first release should expose the policy rather than prematurely hard-code one universal threshold.

### Canonical Document Validation

Exact validation behavior still needs to be finalized for:

* duplicate IDs;
* empty text;
* unsupported metadata;
* generated IDs;
* source handling.

### Artifact Manifest

The manifest must define:

* artifact version;
* corpus checksum;
* model versions;
* taxonomy versions;
* tree configurations;
* label-set versions;
* matrix references;
* dependency references.

### Cache Invalidation

Named cuts and visualization caches need explicit invalidation rules when:

* the parent tree changes;
* a selected label set changes;
* relevant scoring configuration changes.

### Chunking

The first release treats one record as one leaf.

The relationship among source documents, chunks, and leaves is future design work and should not block extraction.

---

## Current Implementation Blockers

The following must be resolved before the standalone refactor is considered complete:

1. Define the canonical document schema and validation rules.
2. Define the persistent artifact manifest.
3. Define stable IDs for documents, trees, nodes, views, taxonomies, and label sets.
4. Specify the mapping between local linkage indexes and canonical node IDs.
5. Define the minimum interface for loading externally generated linkage.
6. Establish compatibility and invalidation rules for cached cuts and label sets.
7. Separate reusable production code from AppWorld-specific adapters and evaluation logic.
8. Identify authoritative notebook and adapter implementations that must be promoted into tested modules.
9. Ensure public defaults are inspectable and reproducible.
10. Ensure all README examples are executable against the standalone package.

The separate `RELEASE_GATE.md` defines the deterministic acceptance tests for these requirements.

---

## Explicitly Out of Scope

For the first standalone release:

* incremental document insertion;
* document deletion;
* document replacement;
* online tree maintenance;
* partial tree rebuild;
* SQL or PostgreSQL dependency;
* distributed processing;
* hosted service deployment;
* multi-user access;
* front-end search and exploration workflows;
* synchronized traversal across multiple trees;
* automatic best-tree selection;
* very large corpus support;
* permanent storage of every generated cut;
* full migration of historical READMEs or planning documents into a wiki.

---

## Stubs for Future Work

The library should preserve clean extension points for future capabilities without partially implementing them now.

### Projection and Grafting

A future:

```python
orchard.transform(new_documents)
```

may:

1. enrich new records;
2. locate appropriate existing tree positions;
3. graft records into the existing hierarchy;
4. modify the Z-array while preserving canonical ordering and structural validity.

The detailed algorithm already exists conceptually in prior work but is outside this release.

For now this capability should either be absent from the public API or explicitly raise `NotImplementedError`.

It must not silently approximate insertion by simply assigning a new record to an existing cluster.

### Corpus Mutation

Future support may include:

* append;
* delete;
* replace;
* partial rebuild;
* label invalidation;
* matrix refresh;
* vector-index refresh.

### Vector Indexing

Embeddings may later be indexed using FAISS or another lightweight local vector structure for:

* hybrid search;
* tree traversal;
* projection;
* grafting;
* candidate-node retrieval.

### Cross-Tree Navigation

A selection made in one tree may later constrain the active corpus represented by another tree.

This belongs to a later query/navigation layer and is not part of the initial standalone extraction.

### Service Wrapper

A future service may wrap the Python library for:

* shared compute;
* job execution;
* persistence;
* concurrency;
* very large corpora.

The service must remain a wrapper around the library rather than redefining the core Orchard implementation.

---

## Refactor Principle

The standalone package should preserve existing functional behavior before optimizing or redesigning implementation details.

The refactor should:

* extract authoritative implementations from notebooks and adapters;
* remove AppWorld phase-number terminology from the public API;
* retain structural validation at artifact boundaries;
* replace experimental and phase-specific tests with a compact set of contract and invariance tests;
* retain tests that protect linkage, identity, membership, cut, subtree, persistence, and visualization invariants;
* remove obsolete planning documents and development artifacts from the package;
* keep AppWorld-specific code outside the Orchard core;
* avoid changing the underlying modeling stack unless required for isolation;
* preserve current working algorithms unless an architectural boundary requires change;
* favor a small public API over merely minimizing the number of source files.

The objective is not simply a smaller repository.

It is a clear distinction between:

* enriching a corpus;
* building a persistent hierarchical schema;
* labeling that schema;
* querying and reshaping it;
* evaluating it in an external test suite.

---

## Working Definition

> **Orchard is a local-first Python library that builds persistent hierarchical schemas over finite corpora and exposes those schemas as dynamically cuttable, prunable, labelable, queryable, and visualizable trees.**

The initial release is successful when Orchard can stand alone as a compact Python dependency, AppWorld can consume it externally, and the deterministic conditions in `RELEASE_GATE.md` pass.
