"""Phase 2 MiniLM transforms and semantic-default wiring (no model download)."""

from __future__ import annotations

import numpy as np
import pytest

from orchard import MissingOptionalDependencyError, OrchardBuilder
from orchard.backends.minilm import (
    MINILM_DIMENSIONS,
    MINILM_MODEL_ID,
    MINILM_REVISION,
    SENTENCE_CONFIG,
    TRANSFORM_CONFIG,
    MiniLMEmbeddingBackend,
    apply_description_transform,
    embedding_cache_key,
    mapped_cosine,
    minilm_similarity_matrix,
    minilm_snapshot_available,
    raw_cosine,
)
from orchard.backends.tfidf import TfidfEmbeddingBackend
from orchard.fixtures import load_documents


def _fake_vectors(n_docs: int, dim: int = 8, seed: int = 20260725) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n_docs, dim))


def test_centering_then_signed_map_on_injected_vectors() -> None:
    raw = np.array(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 0.0]],
        dtype=np.float64,
    )
    centered = apply_description_transform(raw, "centered")
    np.testing.assert_allclose(centered, raw - raw.mean(axis=0))
    assert apply_description_transform(raw, "raw") is not None
    np.testing.assert_allclose(apply_description_transform(raw, "raw"), raw)

    raw_s = raw_cosine(centered)
    mapped = mapped_cosine(raw_s)
    np.testing.assert_allclose(mapped, np.clip((raw_s + 1.0) / 2.0, 0.0, 1.0))
    np.testing.assert_allclose(np.diag(mapped), 1.0)
    fused = minilm_similarity_matrix(raw, transform="centered")
    np.testing.assert_allclose(fused, mapped)


def test_whitening_uses_sample_covariance_and_ridge() -> None:
    raw = _fake_vectors(8, dim=6)
    centered = raw - raw.mean(axis=0)
    covariance = centered.T @ centered / (len(raw) - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    maximum = float(eigenvalues[0])
    floor = maximum * float(TRANSFORM_CONFIG["whitening_relative_eigenvalue_floor"])
    retained = eigenvalues > floor
    ridge = maximum * float(TRANSFORM_CONFIG["whitening_ridge_fraction"])
    expected = (centered @ eigenvectors[:, retained]) / np.sqrt(
        eigenvalues[retained] + ridge
    )
    actual = apply_description_transform(raw, "whitened")
    np.testing.assert_allclose(actual, expected)
    assert TRANSFORM_CONFIG["whitening_relative_eigenvalue_floor"] == 1e-6
    assert TRANSFORM_CONFIG["whitening_ridge_fraction"] == 1e-4


def test_centered_pc3_drops_three_principal_components() -> None:
    raw = _fake_vectors(10, dim=8)
    centered = raw - raw.mean(axis=0)
    covariance = centered.T @ centered / (len(raw) - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, order]
    removed = int(TRANSFORM_CONFIG["pca_components_removed"])
    assert removed == 3
    expected = centered - (centered @ eigenvectors[:, :removed]) @ eigenvectors[
        :, :removed
    ].T
    actual = apply_description_transform(raw, "centered_pc3")
    np.testing.assert_allclose(actual, expected)


def test_cache_key_includes_model_revision_texts_and_config() -> None:
    texts = ["alpha", "bravo"]
    key_a = embedding_cache_key(texts)
    key_b = embedding_cache_key(texts, revision="other-revision")
    key_c = embedding_cache_key(["alpha", "charlie"])
    key_d = embedding_cache_key(
        texts,
        configuration={**SENTENCE_CONFIG, "max_length": 128},
    )
    assert key_a != key_b
    assert key_a != key_c
    assert key_a != key_d
    assert key_a == embedding_cache_key(texts, model_id=MINILM_MODEL_ID)


def test_injected_minilm_semantic_build_uses_fused_default() -> None:
    documents = load_documents()
    vectors = _fake_vectors(len(documents), dim=8)
    backend = MiniLMEmbeddingBackend(encode_fn=lambda _texts: vectors)
    builder = OrchardBuilder(taxonomies=[], embedding_backend=backend)
    orchard = builder.build(documents)
    assert orchard.tree_ids == ("semantic",)
    assert orchard.tree("semantic").leaf_count == len(documents)
    params = builder.get_params()
    assert params["fusion_mode"] == "variance_calibrated"
    assert params["profiles"]["semantic"]["fusion_mode"] == "variance_calibrated"
    assert params["profiles"]["semantic"]["weights"] == {
        "description_minilm_centered_cosine": 0.66,
        "tfidf_cosine": 0.34,
    }
    assert params["active_layers"]["semantic"] == [
        "description_minilm_centered_cosine",
        "tfidf_cosine",
    ]
    assert params["embedding_backend"] == "MiniLMEmbeddingBackend"
    assert params["model_id"] == MINILM_MODEL_ID
    assert params["model_revision"] == MINILM_REVISION
    assert params["pooling"] == SENTENCE_CONFIG["pooling"]
    assert params["description_transform"] == "centered"
    assert params["allow_offline_fallback"] is False
    assert params["offline_fallback"] is False


def test_explicit_tfidf_backend_stays_tfidf_only() -> None:
    documents = load_documents()
    builder = OrchardBuilder(
        taxonomies=[],
        embedding_backend=TfidfEmbeddingBackend(),
    )
    orchard = builder.build(documents)
    assert orchard.tree_ids == ("semantic",)
    params = builder.get_params()
    assert params["embedding_backend"] == "TfidfEmbeddingBackend"
    assert params["profiles"]["semantic"]["weights"] == {"tfidf_cosine": 1.0}
    assert params["model_id"] is None
    assert params["offline_fallback"] is False


def test_missing_extra_without_fallback_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "orchard.builder.embeddings_extra_available",
        lambda: False,
    )
    with pytest.raises(MissingOptionalDependencyError, match="orchard\\[embeddings\\]"):
        OrchardBuilder(taxonomies=[])


def test_offline_fallback_builds_tfidf_and_records_provenance(
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
    assert params["allow_offline_fallback"] is True
    assert params["embedding_backend"] == "TfidfEmbeddingBackend"
    assert params["profiles"]["semantic"]["weights"] == {"tfidf_cosine": 1.0}
    assert params["model_id"] is None
    dumped = str(params).lower()
    assert "offline_fallback" in dumped
    assert "minilm" not in dumped
    assert MINILM_MODEL_ID not in str(params)


@pytest.mark.optional_model
def test_live_minilm_encode_optional(tmp_path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    if not minilm_snapshot_available():
        pytest.skip("MiniLM revision snapshot is not present locally")
    backend = MiniLMEmbeddingBackend(cache_dir=tmp_path)
    features = backend.encode(["hello world", "another sentence"])
    assert features.shape == (2, MINILM_DIMENSIONS)
    assert np.isfinite(features).all()
