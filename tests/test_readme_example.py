"""Phase 4: README minimal example runs as an automated test."""

from __future__ import annotations

from orchard import OrchardBuilder


def test_readme_minimal_example() -> None:
    orchard = OrchardBuilder().build(
        [
            "Schedule a calendar reminder for tomorrow morning.",
            "Send an email summary of the weekly status notes.",
            "Create a task to review the draft proposal.",
            "Find documents mentioning quarterly budget planning.",
        ]
    )
    assert orchard.tree_ids == ("domain", "function")
    domain = orchard.tree("domain")
    assert domain.leaf_count == 4


def test_readme_semantic_corner() -> None:
    documents = [
        "Schedule a calendar reminder for tomorrow morning.",
        "Send an email summary of the weekly status notes.",
        "Create a task to review the draft proposal.",
        "Find documents mentioning quarterly budget planning.",
    ]
    orchard = OrchardBuilder(taxonomies=[]).build(documents)
    semantic = orchard.tree("semantic")
    assert semantic.tree_id == "semantic"
    assert semantic.leaf_count == 4
