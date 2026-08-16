"""Phase 2: no-taxonomy build → semantic tree (MiniLM default / TF-IDF opt-in)."""

from __future__ import annotations

import numpy as np
import pytest

from orchard import MissingOptionalDependencyError, OrchardBuilder
from orchard.backends.minilm import (
    MINILM_MODEL_ID,
    MINILM_REVISION,
    SENTENCE_CONFIG,
    MiniLMEmbeddingBackend,
)
from orchard.backends.similarity import cosine_matrix, linkage_from_similarity
from orchard.backends.tfidf import TfidfEmbeddingBackend
from orchard.fixtures import load_documents


def _injected_minilm(n_docs: int) -> MiniLMEmbeddingBackend:
    rng = np.random.default_rng(20260725)
    vectors = rng.normal(size=(n_docs, 8))
    return MiniLMEmbeddingBackend(encode_fn=lambda _texts: vectors)


def test_embeddings_extra_selects_minilm_default_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "orchard.builder.embeddings_extra_available",
        lambda: True,
    )
    builder = OrchardBuilder(taxonomies=[])
    assert isinstance(builder.embedding_backend, MiniLMEmbeddingBackend)
    params = builder.get_params()
    assert params["profiles"]["semantic"]["weights"] == {
        "description_minilm_centered_cosine": 0.66,
        "tfidf_cosine": 0.34,
    }
    assert params["fusion_mode"] == "variance_calibrated"
    assert params["offline_fallback"] is False


def test_tiny_corpus_injected_minilm_builds_fused_semantic_tree() -> None:
    documents = load_documents()
    builder = OrchardBuilder(
        taxonomies=[],
        embedding_backend=_injected_minilm(len(documents)),
    )
    orchard = builder.build(documents)

    assert orchard.tree_ids == ("semantic",)
    tree = orchard.tree("semantic")
    assert tree.leaf_count == len(documents)
    assert set(tree.item_ids) == {doc.item_id for doc in documents}
    assert tree.linkage.shape == (len(documents) - 1, 4)
    params = builder.get_params()
    assert params["embedding_backend"] == "MiniLMEmbeddingBackend"
    assert params["fusion_mode"] == "variance_calibrated"
    assert params["profiles"]["semantic"]["weights"] == {
        "description_minilm_centered_cosine": 0.66,
        "tfidf_cosine": 0.34,
    }
    assert params["model_id"] == MINILM_MODEL_ID
    assert params["model_revision"] == MINILM_REVISION
    assert params["pooling"] == SENTENCE_CONFIG["pooling"]
    assert params["description_transform"] == "centered"
    assert params["allow_offline_fallback"] is False


def test_explicit_tfidf_backend_matches_today_d_equals_one_minus_s() -> None:
    documents = load_documents()
    builder = OrchardBuilder(
        taxonomies=[],
        embedding_backend=TfidfEmbeddingBackend(),
    )
    orchard = builder.build(documents)
    backend = TfidfEmbeddingBackend()
    features = backend.encode([document.text for document in orchard.documents])
    similarity = cosine_matrix(features, signed=False)
    expected = linkage_from_similarity(similarity, method="average")
    np.testing.assert_array_equal(orchard.tree("semantic").linkage, expected)
    params = builder.get_params()
    assert params["embedding_backend"] == "TfidfEmbeddingBackend"
    assert params["profiles"]["semantic"]["weights"] == {"tfidf_cosine": 1.0}
    assert params["offline_fallback"] is False


def test_allow_offline_fallback_records_tfidf_not_minilm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "orchard.builder.embeddings_extra_available",
        lambda: False,
    )
    builder = OrchardBuilder(taxonomies=[], allow_offline_fallback=True)
    orchard = builder.build(load_documents())
    assert orchard.tree_ids == ("semantic",)
    params = builder.get_params()
    assert params["offline_fallback"] is True
    assert params["embedding_backend"] == "TfidfEmbeddingBackend"
    assert params["profiles"]["semantic"]["weights"] == {"tfidf_cosine": 1.0}
    assert params["model_id"] is None
    assert "minilm" not in str(params).lower()


def test_missing_extra_default_backend_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "orchard.builder.embeddings_extra_available",
        lambda: False,
    )
    with pytest.raises(MissingOptionalDependencyError, match="allow_offline_fallback"):
        OrchardBuilder(taxonomies=[])


def test_builder_accepts_raw_strings() -> None:
    orchard = OrchardBuilder(
        taxonomies=[],
        embedding_backend=TfidfEmbeddingBackend(),
    ).build(
        [
            "alpha calendar reminder scheduling",
            "bravo email messaging notes",
            "charlie task review proposal",
            "delta search budget documents",
        ]
    )
    assert orchard.tree_ids == ("semantic",)
    assert len(orchard.documents) == 4
