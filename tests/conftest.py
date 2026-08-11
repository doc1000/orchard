"""Shared fixtures for Orchard tests."""

from __future__ import annotations

import pytest

from orchard import Document, Orchard, Tree
from orchard.fixtures import load_documents, load_sample_linkage


@pytest.fixture
def documents() -> list[Document]:
    return load_documents()


@pytest.fixture
def sample_tree(documents: list[Document]) -> Tree:
    payload = load_sample_linkage()
    return Tree.from_linkage(
        payload["linkage"],
        item_ids=payload["leaf_ids"],
        tree_id="semantic",
        documents=documents,
    )


@pytest.fixture
def alt_tree(documents: list[Document]) -> Tree:
    """Second linkage on the same corpus (different merge order/distances)."""
    linkage = [
        [1.0, 2.0, 0.15, 2.0],
        [0.0, 3.0, 0.25, 2.0],
        [4.0, 5.0, 0.60, 4.0],
    ]
    item_ids = [doc.item_id for doc in documents]
    return Tree.from_linkage(
        linkage,
        item_ids=item_ids,
        tree_id="custom",
        documents=documents,
    )


@pytest.fixture
def multi_orchard(
    documents: list[Document],
    sample_tree: Tree,
    alt_tree: Tree,
) -> Orchard:
    return Orchard.from_trees(
        documents=documents,
        trees={"semantic": sample_tree, "custom": alt_tree},
    )
