"""Phase 2: no-taxonomy build → semantic tree (offline, no API keys)."""

from __future__ import annotations

from orchard import OrchardBuilder
from orchard.fixtures import load_documents


def test_tiny_corpus_no_taxonomies_builds_semantic_tree() -> None:
    documents = load_documents()
    builder = OrchardBuilder(taxonomies=[])
    orchard = builder.build(documents)

    assert orchard.tree_ids == ("semantic",)
    tree = orchard.tree("semantic")
    assert tree.leaf_count == len(documents)
    assert set(tree.item_ids) == {doc.item_id for doc in documents}
    assert tree.linkage.shape == (len(documents) - 1, 4)
    assert builder.get_params()["embedding_backend"] == "TfidfEmbeddingBackend"
    assert "fused" not in str(builder.get_params()).lower()


def test_builder_accepts_raw_strings() -> None:
    orchard = OrchardBuilder().build(
        [
            "alpha calendar reminder scheduling",
            "bravo email messaging notes",
            "charlie task review proposal",
            "delta search budget documents",
        ]
    )
    assert orchard.tree_ids == ("semantic",)
    assert len(orchard.documents) == 4
