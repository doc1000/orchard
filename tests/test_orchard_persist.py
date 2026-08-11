"""Phase 1 multi-tree save/load identity tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from orchard import CorpusMutationUnsupportedError, Orchard, Tree


def test_save_load_preserves_docs_and_canonical_ids(
    multi_orchard: Orchard,
    tmp_path: Path,
) -> None:
    out = multi_orchard.save(tmp_path / "orchard_art")
    loaded = Orchard.load(out)

    assert loaded.tree_ids == ("custom", "semantic")
    assert [doc.item_id for doc in loaded.documents] == [
        doc.item_id for doc in multi_orchard.documents
    ]
    for name in loaded.tree_ids:
        original = multi_orchard.tree(name)
        restored = loaded.tree(name)
        assert restored.root_node_id == original.root_node_id
        assert set(restored.nodes) == set(original.nodes)
        assert restored.item_ids == original.item_ids
        assert restored.linkage.tolist() == original.linkage.tolist()


def test_orchard_tree_accessor_and_mutation_unsupported(
    multi_orchard: Orchard,
) -> None:
    assert isinstance(multi_orchard.tree("semantic"), Tree)
    with pytest.raises(CorpusMutationUnsupportedError):
        multi_orchard.add_documents([])
    with pytest.raises(CorpusMutationUnsupportedError):
        multi_orchard.remove_documents([])
