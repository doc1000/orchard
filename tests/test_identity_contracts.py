"""Phase 0 identity and uniqueness contract tests."""

from __future__ import annotations

import pytest

from orchard import (
    Document,
    DuplicateItemIdError,
    InvalidIdentityError,
    InvalidLinkageError,
    assign_canonical_ids,
    canonical_node_id,
    ensure_unique_item_ids,
    generate_item_id,
    membership_hash,
    validate_tree_id,
)
from orchard.identity import (
    internal_scipy_index,
    leaf_scipy_index,
    root_scipy_index,
)


def test_document_requires_text_and_defaults() -> None:
    doc = Document(text="hello world")
    assert doc.item_id == generate_item_id("hello world")
    assert doc.title == ""
    assert doc.metadata == {}
    assert doc.source is None


def test_document_item_ids_are_unique_helper() -> None:
    docs = [
        Document(text="a", item_id="a"),
        Document(text="b", item_id="b"),
    ]
    assert ensure_unique_item_ids(d.item_id for d in docs) == ("a", "b")
    with pytest.raises(DuplicateItemIdError):
        ensure_unique_item_ids(["a", "a"])


def test_canonical_node_id_is_membership_stable() -> None:
    assert canonical_node_id(["b", "a"]) == canonical_node_id(["a", "b"])
    assert membership_hash(["a"]) != membership_hash(["a", "b"])
    assert canonical_node_id(["a"]).startswith("member_")


def test_tree_id_validation() -> None:
    assert validate_tree_id("semantic") == "semantic"
    assert validate_tree_id("domain") == "domain"
    with pytest.raises(InvalidIdentityError):
        validate_tree_id("")
    with pytest.raises(InvalidIdentityError):
        validate_tree_id("1bad")


def test_scipy_index_conventions() -> None:
    assert leaf_scipy_index(0) == 0
    assert internal_scipy_index(4, 0) == 4
    assert internal_scipy_index(4, 2) == 6
    assert root_scipy_index(4) == 6
    assert root_scipy_index(1) == 0


def test_assign_canonical_ids_for_sample_z() -> None:
    leaf_ids = ["alpha", "bravo", "charlie", "delta"]
    linkage = [
        [0.0, 1.0, 0.10, 2.0],
        [2.0, 3.0, 0.20, 2.0],
        [4.0, 5.0, 0.50, 4.0],
    ]
    identity = assign_canonical_ids(linkage, leaf_ids)
    assert identity.root_scipy_index == 6
    assert identity.scipy_to_canonical[0] == canonical_node_id(["alpha"])
    assert identity.scipy_to_canonical[4] == canonical_node_id(["alpha", "bravo"])
    assert identity.scipy_to_canonical[5] == canonical_node_id(["charlie", "delta"])
    assert identity.root_canonical_id == canonical_node_id(leaf_ids)
    assert set(identity.memberships[6]) == set(leaf_ids)


def test_assign_canonical_ids_rejects_bad_shape() -> None:
    with pytest.raises(InvalidLinkageError):
        assign_canonical_ids([[0.0, 1.0, 0.1, 2.0]], ["a", "b", "c"])
