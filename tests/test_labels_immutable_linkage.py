"""Phase 3: intrinsic/imported labels never mutate linkage; no contrastive API."""

from __future__ import annotations

import orchard
from orchard import import_labels, label_intrinsic


def test_intrinsic_and_imported_labels_do_not_mutate_linkage(sample_tree) -> None:
    before = sample_tree.linkage_fingerprint()
    intrinsic = label_intrinsic(sample_tree, name="intrinsic_v1")
    assert sample_tree.linkage_fingerprint() == before
    assert sample_tree.root_node_id in intrinsic
    assert sample_tree.active_label_set == "intrinsic_v1"

    import_labels(
        sample_tree,
        {sample_tree.root_node_id: "User root"},
        name="imported",
    )
    assert sample_tree.linkage_fingerprint() == before
    assert sample_tree.labels["imported"][sample_tree.root_node_id] == "User root"

    sample_tree.use_labels("intrinsic_v1")
    assert sample_tree.linkage_fingerprint() == before
    sample_tree.use_labels("imported")
    assert sample_tree.linkage_fingerprint() == before


def test_contrastive_absent_from_public_api() -> None:
    assert "contrastive" not in orchard.__all__
    assert not hasattr(orchard, "label_contrastive")
    assert not hasattr(orchard, "ContrastiveLabeler")
