"""Canonical binary Tree built from SciPy linkage."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from orchard.document import Document
from orchard.exceptions import (
    InvalidIdentityError,
    InvalidLinkageError,
    UnknownLabelSetError,
    UnknownNodeError,
)
from orchard.identity import (
    assign_canonical_ids,
    ensure_unique_item_ids,
    membership_hash,
    validate_tree_id,
)
from orchard.schemas import (
    CANONICAL_TREE_SCHEMA_VERSION,
    LABEL_SET_SCHEMA_VERSION,
    LINKAGE_SCHEMA_VERSION,
)


def build_canonical_nodes(
    linkage: np.ndarray | Sequence[Sequence[float]],
    item_ids: Sequence[str],
) -> dict[str, Any]:
    """Build the flat canonical node registry from Z + ordered leaf IDs."""
    identity = assign_canonical_ids(linkage, item_ids)
    ordered = identity.leaf_ids
    n = len(ordered)
    z = np.asarray(linkage, dtype=float)
    nodes: dict[str, dict[str, Any]] = {}

    for index, item_id in enumerate(ordered):
        node_id = identity.scipy_to_canonical[index]
        members = list(identity.memberships[index])
        nodes[node_id] = {
            "node_id": node_id,
            "kind": "leaf",
            "item_id": item_id,
            "children": [],
            "descendant_count": 1,
            "linkage_distance": 0.0,
            "membership_hash": membership_hash(members),
            "descendant_item_ids": members,
        }

    if n == 1:
        return {
            "schema_version": CANONICAL_TREE_SCHEMA_VERSION,
            "root_node_id": identity.root_canonical_id,
            "leaf_count": 1,
            "internal_node_count": 0,
            "nodes": nodes,
        }

    for row_index, row in enumerate(z):
        scipy_id = n + row_index
        left = int(row[0])
        right = int(row[1])
        node_id = identity.scipy_to_canonical[scipy_id]
        members = list(identity.memberships[scipy_id])
        left_id = identity.scipy_to_canonical[left]
        right_id = identity.scipy_to_canonical[right]
        nodes[node_id] = {
            "node_id": node_id,
            "kind": "internal",
            "children": [left_id, right_id],
            "descendant_count": int(row[3]),
            "linkage_distance": float(row[2]),
            "membership_hash": membership_hash(members),
            "descendant_item_ids": members,
            "scipy_node_id": scipy_id,
            "linkage_row": row_index,
        }

    return {
        "schema_version": CANONICAL_TREE_SCHEMA_VERSION,
        "root_node_id": identity.root_canonical_id,
        "leaf_count": n,
        "internal_node_count": n - 1,
        "nodes": nodes,
    }


def validate_canonical_tree(
    tree: Mapping[str, Any],
    expected_ids: Sequence[str],
) -> None:
    """Validate membership and count invariants on a canonical tree dict."""
    if tree.get("schema_version") != CANONICAL_TREE_SCHEMA_VERSION:
        raise InvalidIdentityError("unsupported canonical tree schema")
    nodes = tree["nodes"]
    root = nodes[tree["root_node_id"]]
    leaves = [node for node in nodes.values() if node["kind"] == "leaf"]
    internal = [node for node in nodes.values() if node["kind"] == "internal"]
    expected = ensure_unique_item_ids(expected_ids)
    if len(leaves) != len(expected) or len(internal) != max(0, len(expected) - 1):
        raise InvalidIdentityError("canonical node counts are invalid")
    leaf_ids = [node["item_id"] for node in leaves]
    if len(leaf_ids) != len(set(leaf_ids)) or set(leaf_ids) != set(expected):
        raise InvalidIdentityError("canonical leaves are incomplete or duplicated")
    for node in nodes.values():
        if node["membership_hash"] != membership_hash(node["descendant_item_ids"]):
            raise InvalidIdentityError("membership identity is invalid")
        if node["descendant_count"] != len(node["descendant_item_ids"]):
            raise InvalidIdentityError("descendant count is invalid")
        if node["kind"] == "internal":
            left, right = (nodes[item] for item in node["children"])
            union = sorted(
                left["descendant_item_ids"] + right["descendant_item_ids"]
            )
            if union != node["descendant_item_ids"]:
                raise InvalidIdentityError(
                    "parent membership is not the exact child union"
                )
    if root["descendant_item_ids"] != sorted(expected):
        raise InvalidIdentityError("root membership is incomplete")


@dataclass
class Tree:
    """Canonical binary hierarchy with optional named label overlays."""

    tree_id: str
    linkage: np.ndarray
    item_ids: tuple[str, ...]
    nodes: dict[str, dict[str, Any]]
    root_node_id: str
    labels: dict[str, dict[str, str]] = field(default_factory=dict)
    active_label_set: str | None = None
    method: str = "average"
    documents_by_id: dict[str, Document] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.tree_id = validate_tree_id(self.tree_id)
        self.item_ids = ensure_unique_item_ids(self.item_ids)
        self.linkage = np.asarray(self.linkage, dtype=float)
        payload = {
            "schema_version": CANONICAL_TREE_SCHEMA_VERSION,
            "tree_id": self.tree_id,
            "root_node_id": self.root_node_id,
            "leaf_count": len(self.item_ids),
            "internal_node_count": max(0, len(self.item_ids) - 1),
            "nodes": self.nodes,
        }
        validate_canonical_tree(payload, self.item_ids)

    @classmethod
    def from_linkage(
        cls,
        linkage: np.ndarray | Sequence[Sequence[float]],
        *,
        item_ids: Sequence[str],
        tree_id: str = "tree",
        method: str = "average",
        documents: Sequence[Document] | None = None,
        labels: Mapping[str, Mapping[str, str]] | None = None,
        active_label_set: str | None = None,
    ) -> Tree:
        ordered = ensure_unique_item_ids(item_ids)
        built = build_canonical_nodes(linkage, ordered)
        docs_by_id: dict[str, Document] = {}
        if documents is not None:
            for doc in documents:
                if doc.item_id in ordered:
                    docs_by_id[doc.item_id] = doc
        return cls(
            tree_id=tree_id,
            linkage=np.asarray(linkage, dtype=float),
            item_ids=ordered,
            nodes=built["nodes"],
            root_node_id=built["root_node_id"],
            labels={name: dict(mapping) for name, mapping in (labels or {}).items()},
            active_label_set=active_label_set,
            method=method,
            documents_by_id=docs_by_id,
        )

    @property
    def leaf_count(self) -> int:
        return len(self.item_ids)

    def node(self, node_id: str) -> dict[str, Any]:
        try:
            return self.nodes[node_id]
        except KeyError as exc:
            raise UnknownNodeError(node_id) from exc

    def linkage_payload(self) -> dict[str, Any]:
        return {
            "schema_version": LINKAGE_SCHEMA_VERSION,
            "tree_id": self.tree_id,
            "method": self.method,
            "item_ids": list(self.item_ids),
            "z_matrix": self.linkage.tolist(),
        }

    def to_canonical_dict(self) -> dict[str, Any]:
        return {
            "schema_version": CANONICAL_TREE_SCHEMA_VERSION,
            "tree_id": self.tree_id,
            "root_node_id": self.root_node_id,
            "leaf_count": self.leaf_count,
            "internal_node_count": max(0, self.leaf_count - 1),
            "nodes": deepcopy(self.nodes),
        }

    def set_labels(
        self,
        name: str,
        mapping: Mapping[str, str],
        *,
        make_active: bool = True,
    ) -> None:
        """Store or replace a named label set. Never mutates linkage."""
        if not name or not str(name).strip():
            raise InvalidIdentityError("label set name must be non-empty")
        cleaned: dict[str, str] = {}
        for node_id, label in mapping.items():
            if node_id not in self.nodes:
                raise UnknownNodeError(node_id)
            cleaned[node_id] = str(label)
        self.labels[str(name)] = cleaned
        if make_active:
            self.active_label_set = str(name)

    def use_labels(self, name: str | None) -> None:
        if name is None:
            self.active_label_set = None
            return
        if name not in self.labels:
            raise UnknownLabelSetError(name)
        self.active_label_set = name

    def label_for(self, node_id: str, label_set: str | None = None) -> str | None:
        name = self.active_label_set if label_set is None else label_set
        if name is None:
            return None
        if name not in self.labels:
            raise UnknownLabelSetError(name)
        return self.labels[name].get(node_id)

    def linkage_fingerprint(self) -> tuple[str, ...]:
        """Stable fingerprint of linkage + leaf order for immutability checks."""
        flat = ",".join(f"{value:.12g}" for value in self.linkage.ravel().tolist())
        return self.item_ids + (flat,)

    def subtree(self, node_id: str, *, tree_id: str | None = None) -> Tree:
        """Return a reduced Tree for one canonical branch (identity-preserving)."""
        from orchard.cuts import linkage_for_canonical_subtree

        node = self.node(node_id)
        if node["kind"] == "leaf":
            item_id = node["item_id"]
            return Tree.from_linkage(
                np.zeros((0, 4)),
                item_ids=[item_id],
                tree_id=tree_id or f"{self.tree_id}_subtree",
                method=self.method,
                documents=[self.documents_by_id[item_id]]
                if item_id in self.documents_by_id
                else None,
            )
        z_local, local_ids = linkage_for_canonical_subtree(
            self.to_canonical_dict(),
            node_id,
            linkage=self.linkage_payload(),
        )
        # Canonical IDs must match the source branch for shared memberships.
        built = Tree.from_linkage(
            z_local,
            item_ids=local_ids,
            tree_id=tree_id or f"{self.tree_id}_subtree",
            method=self.method,
            documents=[
                self.documents_by_id[item_id]
                for item_id in local_ids
                if item_id in self.documents_by_id
            ],
        )
        if built.root_node_id != node_id:
            raise InvalidLinkageError(
                "subtree root canonical ID does not match source node"
            )
        # Carry overlapping label entries without touching linkage.
        for name, mapping in self.labels.items():
            subset = {
                key: value
                for key, value in mapping.items()
                if key in built.nodes
            }
            if subset:
                built.labels[name] = subset
        built.active_label_set = self.active_label_set
        return built

    @classmethod
    def from_persisted(
        cls,
        *,
        canonical: Mapping[str, Any],
        linkage: Mapping[str, Any],
        labels: Mapping[str, Mapping[str, str]] | None = None,
        active_label_set: str | None = None,
        documents: Sequence[Document] | None = None,
    ) -> Tree:
        if linkage.get("schema_version") != LINKAGE_SCHEMA_VERSION:
            raise InvalidLinkageError("unsupported linkage schema")
        item_ids = list(linkage["item_ids"])
        tree = cls.from_linkage(
            linkage["z_matrix"],
            item_ids=item_ids,
            tree_id=str(canonical.get("tree_id") or linkage.get("tree_id") or "tree"),
            method=str(linkage.get("method") or "average"),
            documents=documents,
            labels=labels,
            active_label_set=active_label_set,
        )
        if tree.root_node_id != canonical["root_node_id"]:
            raise InvalidIdentityError("persisted root_node_id mismatch")
        if set(tree.nodes) != set(canonical["nodes"]):
            raise InvalidIdentityError("persisted node registry mismatch")
        return tree


def default_node_label(tree: Tree, node_id: str, label_set: str | None = None) -> str:
    """Resolve a display label: named set, else item_id / short node id."""
    named = tree.label_for(node_id, label_set=label_set)
    if named is not None:
        return named
    node = tree.node(node_id)
    if node["kind"] == "leaf":
        return str(node["item_id"])
    return node_id[:18]
