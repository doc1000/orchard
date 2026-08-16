"""Phase 4: README minimal example runs as an automated test."""

from __future__ import annotations

import pytest

from orchard import OrchardBuilder
from orchard.backends.tfidf import TfidfEmbeddingBackend


def test_readme_minimal_example(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "orchard.builder.taxonomy_ml_extra_available",
        lambda: False,
    )
    monkeypatch.setattr(
        "orchard.taxonomy.taxonomy_ml_extra_available",
        lambda: False,
    )
    monkeypatch.setattr(
        "orchard.builder.embeddings_extra_available",
        lambda: False,
    )
    orchard = OrchardBuilder(allow_offline_fallback=True).build(
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
    orchard = OrchardBuilder(
        taxonomies=[],
        embedding_backend=TfidfEmbeddingBackend(),
    ).build(documents)
    semantic = orchard.tree("semantic")
    assert semantic.tree_id == "semantic"
    assert semantic.leaf_count == 4
