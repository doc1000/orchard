"""External linkage path: cut / plot / persist / import labels."""

from __future__ import annotations

from pathlib import Path

from orchard import (
    Orchard,
    Tree,
    build_dynamic_cut,
    cut_to_plotly_structure,
    orchard_plotly_structures,
    validate_dynamic_cut,
    validate_plotly_structure,
)


def test_external_linkage_cut_plot_persist_labels(
    documents,
    sample_tree: Tree,
    alt_tree: Tree,
    tmp_path: Path,
) -> None:
    # Import labels without touching linkage.
    before = sample_tree.linkage_fingerprint()
    sample_tree.set_labels(
        "user_v1",
        {sample_tree.root_node_id: "All items"},
    )
    sample_tree.set_labels(
        "user_v2",
        {sample_tree.root_node_id: "Everything"},
        make_active=True,
    )
    assert sample_tree.linkage_fingerprint() == before
    assert sample_tree.active_label_set == "user_v2"

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

    structure = cut_to_plotly_structure(
        sample_tree,
        cut,
        cut_id="semantic_cut",
        label_set="user_v2",
    )
    report = validate_plotly_structure(
        structure,
        sample_tree,
        cut,
        cut_id="semantic_cut",
    )
    assert report["status"] == "passed", report["errors"]
    assert structure["root"]["label"] == "Everything"

    orchard = Orchard.from_trees(
        documents=documents,
        trees={"semantic": sample_tree, "custom": alt_tree},
    )
    cut_b = build_dynamic_cut(
        alt_tree,
        top_criterion="cluster_count",
        cluster_count=2,
        min_width=1,
        max_width=4,
        target_width=2,
        max_depth=1,
    )
    multi = orchard_plotly_structures(
        orchard,
        {"semantic": cut, "custom": cut_b},
    )
    assert set(multi["tree_ids"]) == {"custom", "semantic"}

    saved = orchard.save(tmp_path / "ext")
    loaded = Orchard.load(saved)
    assert loaded.tree("semantic").labels["user_v2"][sample_tree.root_node_id] == (
        "Everything"
    )
    assert loaded.tree("semantic").linkage_fingerprint() == before
