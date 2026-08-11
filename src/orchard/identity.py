"""Shared item IDs, tree IDs, and canonical node identity contracts.

Canonical node IDs follow the reference membership-hash scheme:
``member_<sha256(canonical_json(sorted(member_item_ids)))>``.

SciPy linkage local indexes use the standard convention:
- leaves occupy ``0 .. n-1`` in leaf order;
- linkage row ``r`` creates internal node ``n + r``;
- the root is ``2 * n - 2``.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from orchard.exceptions import (
    DuplicateItemIdError,
    InvalidIdentityError,
    InvalidLinkageError,
)
from orchard.schemas import LINKAGE_INDEX_SCHEMA_VERSION

_TREE_ID_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_\-]*$")


def canonical_json(value: Any) -> str:
    """Deterministic JSON used for membership hashing."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def membership_hash(item_ids: Sequence[str]) -> str:
    """Hash of a node's descendant item-ID set (sorted)."""
    members = sorted(item_ids)
    if len(members) != len(set(members)):
        raise InvalidIdentityError("membership item_ids must be unique")
    for item_id in members:
        if not item_id or not str(item_id).strip():
            raise InvalidIdentityError("membership item_ids must be non-empty")
    return sha256_text(canonical_json(members))


def canonical_node_id(item_ids: Sequence[str]) -> str:
    """Stable canonical node ID derived solely from member item IDs."""
    return f"member_{membership_hash(item_ids)}"


def generate_item_id(text: str) -> str:
    """Generate a stable item_id when the caller does not supply one."""
    digest = sha256_text(text)
    return f"doc_{digest[:16]}"


def validate_tree_id(tree_id: str) -> str:
    """Validate a tree name/id used inside a multi-tree Orchard."""
    if not isinstance(tree_id, str) or not tree_id.strip():
        raise InvalidIdentityError("tree_id must be a non-empty string")
    normalized = tree_id.strip()
    if not _TREE_ID_RE.match(normalized):
        raise InvalidIdentityError(
            "tree_id must start with a letter and contain only "
            "letters, digits, underscore, or hyphen"
        )
    return normalized


def ensure_unique_item_ids(item_ids: Iterable[str]) -> tuple[str, ...]:
    """Return item IDs as a tuple, rejecting duplicates or empties."""
    ordered: list[str] = []
    seen: set[str] = set()
    for raw in item_ids:
        item_id = str(raw).strip() if raw is not None else ""
        if not item_id:
            raise InvalidIdentityError("item_id must be non-empty")
        if item_id in seen:
            raise DuplicateItemIdError(f"duplicate item_id: {item_id}")
        seen.add(item_id)
        ordered.append(item_id)
    return tuple(ordered)


def leaf_scipy_index(leaf_position: int) -> int:
    if leaf_position < 0:
        raise InvalidLinkageError("leaf position must be >= 0")
    return leaf_position


def internal_scipy_index(n_leaves: int, linkage_row: int) -> int:
    if n_leaves < 1:
        raise InvalidLinkageError("n_leaves must be >= 1")
    if linkage_row < 0 or linkage_row >= n_leaves - 1:
        raise InvalidLinkageError("linkage_row out of range for n_leaves")
    return n_leaves + linkage_row


def root_scipy_index(n_leaves: int) -> int:
    if n_leaves < 1:
        raise InvalidLinkageError("n_leaves must be >= 1")
    if n_leaves == 1:
        return 0
    return 2 * n_leaves - 2


@dataclass(frozen=True, slots=True)
class LinkageIdentity:
    """Local SciPy index ↔ canonical node ID assignment for one tree."""

    schema_version: str
    leaf_ids: tuple[str, ...]
    scipy_to_canonical: Mapping[int, str]
    memberships: Mapping[int, tuple[str, ...]]
    root_scipy_index: int
    root_canonical_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "leaf_ids": list(self.leaf_ids),
            "scipy_to_canonical": {
                str(index): node_id
                for index, node_id in sorted(self.scipy_to_canonical.items())
            },
            "memberships": {
                str(index): list(members)
                for index, members in sorted(self.memberships.items())
            },
            "root_scipy_index": self.root_scipy_index,
            "root_canonical_id": self.root_canonical_id,
        }


def assign_canonical_ids(
    linkage: np.ndarray | Sequence[Sequence[float]],
    leaf_ids: Sequence[str],
) -> LinkageIdentity:
    """Map SciPy local indexes to canonical node IDs for a linkage matrix.

    This is the identity contract only: it does not build a full Tree object.
    """
    ordered = ensure_unique_item_ids(leaf_ids)
    n = len(ordered)
    if n == 0:
        raise InvalidLinkageError("leaf_ids must be non-empty")

    z = np.asarray(linkage, dtype=float)
    if n == 1:
        if z.size != 0:
            raise InvalidLinkageError("single-leaf linkage must be empty")
        leaf_id = ordered[0]
        node_id = canonical_node_id((leaf_id,))
        return LinkageIdentity(
            schema_version=LINKAGE_INDEX_SCHEMA_VERSION,
            leaf_ids=ordered,
            scipy_to_canonical={0: node_id},
            memberships={0: (leaf_id,)},
            root_scipy_index=0,
            root_canonical_id=node_id,
        )

    if z.ndim != 2 or z.shape != (n - 1, 4):
        raise InvalidLinkageError(
            f"linkage must have shape ({n - 1}, 4); got {getattr(z, 'shape', None)}"
        )
    if not np.isfinite(z).all():
        raise InvalidLinkageError("linkage contains non-finite values")

    scipy_to_canonical: dict[int, str] = {}
    memberships: dict[int, tuple[str, ...]] = {}
    for index, item_id in enumerate(ordered):
        members = (item_id,)
        node_id = canonical_node_id(members)
        scipy_to_canonical[index] = node_id
        memberships[index] = members

    for row_index, row in enumerate(z):
        left = int(row[0])
        right = int(row[1])
        if left not in memberships or right not in memberships:
            raise InvalidLinkageError(
                f"linkage row {row_index} references unknown indexes "
                f"({left}, {right})"
            )
        if left == right:
            raise InvalidLinkageError(
                f"linkage row {row_index} merges a node with itself"
            )
        members = tuple(sorted(memberships[left] + memberships[right]))
        if len(members) != len(set(members)):
            raise InvalidLinkageError(
                f"linkage row {row_index} produced overlapping memberships"
            )
        scipy_id = internal_scipy_index(n, row_index)
        node_id = canonical_node_id(members)
        scipy_to_canonical[scipy_id] = node_id
        memberships[scipy_id] = members

    root_index = root_scipy_index(n)
    root_id = scipy_to_canonical[root_index]
    if set(memberships[root_index]) != set(ordered):
        raise InvalidLinkageError("root membership does not cover all leaves")

    return LinkageIdentity(
        schema_version=LINKAGE_INDEX_SCHEMA_VERSION,
        leaf_ids=ordered,
        scipy_to_canonical=scipy_to_canonical,
        memberships=memberships,
        root_scipy_index=root_index,
        root_canonical_id=root_id,
    )
