"""Phase 2: one tree per taxonomy; no fused default tree."""

from __future__ import annotations

from orchard import OrchardBuilder, StubTaxonomy
from orchard.fixtures import load_documents


def _fixture_taxonomies() -> list[StubTaxonomy]:
    return [
        StubTaxonomy(
            name="domain",
            label_order=("schedule", "comms", "work", "search"),
            assignments={
                "alpha": {
                    "schedule": 0.7,
                    "comms": 0.1,
                    "work": 0.1,
                    "search": 0.1,
                },
                "bravo": {
                    "schedule": 0.1,
                    "comms": 0.7,
                    "work": 0.1,
                    "search": 0.1,
                },
                "charlie": {
                    "schedule": 0.1,
                    "comms": 0.1,
                    "work": 0.7,
                    "search": 0.1,
                },
                "delta": {
                    "schedule": 0.1,
                    "comms": 0.1,
                    "work": 0.1,
                    "search": 0.7,
                },
            },
        ),
        StubTaxonomy(
            name="function",
            label_order=("create", "send", "find"),
            assignments={
                "alpha": {"create": 0.6, "send": 0.2, "find": 0.2},
                "bravo": {"create": 0.2, "send": 0.6, "find": 0.2},
                "charlie": {"create": 0.6, "send": 0.2, "find": 0.2},
                "delta": {"create": 0.2, "send": 0.2, "find": 0.6},
            },
        ),
    ]


def test_multi_taxonomy_build_produces_distinct_named_trees() -> None:
    documents = load_documents()
    builder = OrchardBuilder(taxonomies=_fixture_taxonomies())
    orchard = builder.build(documents)

    assert orchard.tree_ids == ("domain", "function")
    assert "semantic" not in orchard.trees
    domain = orchard.tree("domain")
    function = orchard.tree("function")
    assert domain.root_node_id != function.root_node_id or (
        domain.linkage.tolist() != function.linkage.tolist()
    )
    assert set(domain.item_ids) == set(function.item_ids)
    params = builder.get_params()
    assert params["taxonomies"] == ["domain", "function"]
    assert params["include_semantic_with_taxonomies"] is False


def test_no_mixed_fused_default_tree_in_api() -> None:
    orchard = OrchardBuilder(taxonomies=_fixture_taxonomies()).build(load_documents())
    assert set(orchard.trees) == {"domain", "function"}
    for name in orchard.tree_ids:
        assert "mixed" not in name
        assert "fused" not in name
