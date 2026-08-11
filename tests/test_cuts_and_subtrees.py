"""Phase 1 cut partition and subtree identity tests."""

from __future__ import annotations

from orchard import (
    Tree,
    build_dynamic_cut,
    cut_partition_item_ids,
    pack_canonical_tree,
    validate_dynamic_cut,
    walk_cut_json,
)


def test_cluster_count_cut_partitions_leaves(sample_tree: Tree) -> None:
    cut = build_dynamic_cut(
        sample_tree,
        top_criterion="cluster_count",
        cluster_count=2,
        min_width=1,
        max_width=4,
        target_width=2,
        max_depth=2,
    )
    validate_dynamic_cut(cut, sample_tree)
    parts = cut_partition_item_ids(cut, sample_tree)
    flat = [item for group in parts for item in group]
    assert sorted(flat) == sorted(sample_tree.item_ids)
    assert len(flat) == len(set(flat))
    assert cut["top_partition_count"] == 2


def test_subtree_preserves_canonical_ids_and_is_recursive(sample_tree: Tree) -> None:
    root_children = sample_tree.nodes[sample_tree.root_node_id]["children"]
    child_id = root_children[0]
    child = sample_tree.node(child_id)
    if child["kind"] == "leaf":
        child_id = root_children[1]
        child = sample_tree.node(child_id)
    assert child["kind"] == "internal"

    sub = sample_tree.subtree(child_id)
    assert sub.root_node_id == child_id
    assert set(sub.item_ids) == set(child["descendant_item_ids"])
    # Recursive usability: cut and further subtree.
    if sub.leaf_count >= 2:
        cut = build_dynamic_cut(
            sub,
            top_criterion="cluster_count",
            cluster_count=2,
            min_width=1,
            max_width=sub.leaf_count,
            target_width=2,
            max_depth=1,
        )
        validate_dynamic_cut(cut, sub)
        # Further subtree of the same root remains identity-stable.
        again = sub.subtree(sub.root_node_id)
        assert again.root_node_id == sub.root_node_id


def test_pack_and_walk_helpers(sample_tree: Tree) -> None:
    packed = pack_canonical_tree(
        sample_tree,
        target_exposed_clusters=2,
        max_depth=3,
    )
    assert packed["actual_exposed_clusters"] >= 1
    assert len(packed["frontier_node_ids"]) == packed["actual_exposed_clusters"]

    cut = build_dynamic_cut(
        sample_tree,
        top_criterion="cluster_count",
        cluster_count=2,
        min_width=1,
        max_width=4,
        target_width=2,
        max_depth=1,
    )
    walked = walk_cut_json(cut, sample_tree)
    assert walked["canonical_node_id"] == sample_tree.root_node_id
    assert "label" in walked
