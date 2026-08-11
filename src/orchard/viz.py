"""Visualization payloads (Plotly-oriented data; no Plotly dependency).

Extracted from tool-tree-demo adapters/plotly_nested_structure.py and generalized
for Orchard trees, cuts, and named label sets.
"""

from __future__ import annotations

from typing import Any, Mapping

from orchard.schemas import (
    PLOTLY_ADAPTER_SCHEMA_VERSION,
    PLOTLY_STRUCTURE_SCHEMA_VERSION,
)
from orchard.tree import Tree, default_node_label


def _output_node(
    *,
    tree: Tree,
    cut_id: str,
    node_id: str,
    parent_id: str,
    level: int,
    label_set: str | None,
) -> dict[str, Any]:
    source = tree.node(node_id)
    is_leaf = source["kind"] == "leaf"
    label = default_node_label(tree, node_id, label_set=label_set)
    return {
        "id": node_id,
        "parent": parent_id,
        "label": label,
        "value": 1 if is_leaf else source["descendant_count"],
        "kind": "document" if is_leaf else "cluster",
        "level": level,
        "tree_id": tree.tree_id,
        "cut_id": cut_id,
        "canonical_node_id": node_id,
        "stable_node_id": node_id,
        "membership_hash": source["membership_hash"],
        "descendant_count": source["descendant_count"],
        "descendant_item_ids": list(source["descendant_item_ids"]),
        "item_id": source.get("item_id"),
        "children": [],
    }


def cut_to_plotly_structure(
    tree: Tree,
    cut: Mapping[str, Any],
    *,
    cut_id: str = "default",
    label_set: str | None = None,
) -> dict[str, Any]:
    """Convert one dynamic cut over a Tree into nested Plotly treemap data."""

    def build(cut_node: Mapping[str, Any], parent_id: str, level: int) -> dict[str, Any]:
        source_id = cut_node["canonical_node_id"]
        result = _output_node(
            tree=tree,
            cut_id=cut_id,
            node_id=source_id,
            parent_id=parent_id,
            level=level,
            label_set=label_set,
        )
        child_items: list[dict[str, Any]] = []
        for leaf_id in cut_node.get("direct_leaf_node_ids", []):
            child_items.append(
                _output_node(
                    tree=tree,
                    cut_id=cut_id,
                    node_id=leaf_id,
                    parent_id=source_id,
                    level=level + 1,
                    label_set=label_set,
                )
            )
        for child in cut_node.get("children", []):
            child_items.append(build(child, source_id, level + 1))
        result["children"] = child_items
        return result

    root = build(cut["root"], "", 0)
    rows: list[dict[str, Any]] = []

    def preorder(node: Mapping[str, Any]) -> None:
        rows.append({key: value for key, value in node.items() if key != "children"})
        for child in node["children"]:
            preorder(child)

    preorder(root)
    return {
        "schema_version": PLOTLY_STRUCTURE_SCHEMA_VERSION,
        "adapter_schema_version": PLOTLY_ADAPTER_SCHEMA_VERSION,
        "tree_id": tree.tree_id,
        "cut_id": cut_id,
        "source_tree_schema_version": tree.to_canonical_dict()["schema_version"],
        "source_cut_schema_version": cut["schema_version"],
        "plotly": {
            "trace_type": "treemap",
            "branchvalues": "total",
            "row_fields": {
                "ids": "id",
                "parents": "parent",
                "labels": "label",
                "values": "value",
            },
        },
        "root": root,
        "rows": rows,
    }


def orchard_plotly_structures(
    orchard: Any,
    cuts_by_tree: Mapping[str, Mapping[str, Any]],
    *,
    label_set: str | None = None,
) -> dict[str, Any]:
    """Build Plotly payloads for multiple named trees in one Orchard."""
    structures = {}
    for tree_id, cut in cuts_by_tree.items():
        tree = orchard.tree(tree_id)
        structures[tree_id] = cut_to_plotly_structure(
            tree,
            cut,
            cut_id=f"{tree_id}_cut",
            label_set=label_set,
        )
    return {
        "schema_version": "orchard_multi_tree_plotly_v1",
        "tree_ids": sorted(structures),
        "structures": structures,
    }


def validate_plotly_structure(
    structure: Mapping[str, Any],
    tree: Tree,
    cut: Mapping[str, Any],
    *,
    cut_id: str,
) -> dict[str, Any]:
    """Return structural coherence diagnostics for one tree/cut payload."""
    errors: list[str] = []
    expected_leaves = tree.nodes[tree.root_node_id]["descendant_item_ids"]
    if structure.get("schema_version") != PLOTLY_STRUCTURE_SCHEMA_VERSION:
        errors.append("unsupported Plotly structure schema")
    if structure.get("tree_id") != tree.tree_id or structure.get("cut_id") != cut_id:
        errors.append("tree/cut identity mismatch")
    rows = structure.get("rows", [])
    row_ids = [row.get("id") for row in rows]
    row_by_id = {row.get("id"): row for row in rows}
    if len(row_ids) != len(set(row_ids)):
        errors.append("Plotly row IDs are not unique")
    parent_references_valid = all(
        row.get("parent") == "" or row.get("parent") in row_by_id for row in rows
    )
    if not parent_references_valid:
        errors.append("Plotly parent reference does not resolve")

    nested: list[Mapping[str, Any]] = []
    stack = [structure.get("root", {})]
    while stack:
        item = stack.pop()
        nested.append(item)
        stack.extend(reversed(item.get("children", [])))

    leaves: list[str] = []
    for item in nested:
        source_id = item.get("canonical_node_id")
        if source_id not in tree.nodes:
            errors.append("canonical node reference is invalid")
            continue
        source = tree.nodes[source_id]
        if item.get("membership_hash") != source["membership_hash"]:
            errors.append("membership hash differs from source tree")
        if item.get("descendant_item_ids") != source["descendant_item_ids"]:
            errors.append("source-to-Plotly membership differs")
        if source["kind"] == "leaf":
            leaves.append(source["item_id"])

    if sorted(leaves) != expected_leaves or len(leaves) != len(set(leaves)):
        errors.append("leaf coverage is incomplete or non-unique")

    # Cut root must match structure root identity.
    if structure.get("root", {}).get("canonical_node_id") != cut["root"][
        "canonical_node_id"
    ]:
        errors.append("structure root does not match cut root")

    # Deduplicate errors while preserving order.
    unique_errors = list(dict.fromkeys(errors))
    return {
        "tree_id": tree.tree_id,
        "cut_id": cut_id,
        "status": "passed" if not unique_errors else "failed",
        "errors": unique_errors,
        "row_count": len(rows),
        "leaf_count": len(leaves),
    }
