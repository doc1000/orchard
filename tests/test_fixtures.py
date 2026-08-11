"""Phase 0 fixture loading tests."""

from __future__ import annotations

from orchard import Document, ensure_unique_item_ids
from orchard.fixtures import (
    load_documents,
    load_sample_linkage,
    load_sample_linkage_identity,
)
from orchard.identity import canonical_node_id


def test_documents_fixture_loads_with_unique_ids() -> None:
    docs = load_documents()
    assert len(docs) == 4
    assert all(isinstance(doc, Document) for doc in docs)
    ids = ensure_unique_item_ids(doc.item_id for doc in docs)
    assert ids == ("alpha", "bravo", "charlie", "delta")
    assert all(doc.text for doc in docs)


def test_sample_linkage_fixture_matches_identity_contract() -> None:
    payload = load_sample_linkage()
    assert payload["tree_id"] == "semantic"
    assert list(payload["leaf_ids"]) == ["alpha", "bravo", "charlie", "delta"]
    assert payload["linkage"].shape == (3, 4)

    identity = load_sample_linkage_identity()
    assert identity.leaf_ids == tuple(payload["leaf_ids"])
    assert identity.root_canonical_id == canonical_node_id(payload["leaf_ids"])
    assert len(identity.scipy_to_canonical) == 7  # 4 leaves + 3 internals
