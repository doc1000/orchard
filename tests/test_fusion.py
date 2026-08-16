"""Phase 1 fusion engine and layer registry (CI: no GPU, no downloads)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from orchard import InvalidFusionError, OrchardBuilder, StubTaxonomy
from orchard.backends.fusion import (
    SimilarityProfile,
    fuse_to_dissimilarity,
    raw_convex_fusion,
    variance_calibrated_fusion,
)
from orchard.backends.layers import LayerRegistry, MatrixLayer, TfidfCosineLayer
from orchard.backends.similarity import (
    cosine_matrix,
    jensen_shannon_matrix,
    linkage_from_dissimilarity,
    linkage_from_similarity,
)
from orchard.backends.tfidf import TfidfEmbeddingBackend
from orchard.fixtures import load_documents
from orchard.taxonomy import default_taxonomies


def _symmetric(off_diag: float) -> np.ndarray:
    return np.array([[1.0, off_diag], [off_diag, 1.0]], dtype=np.float64)


def test_raw_convex_2x2_weighted_sum() -> None:
    matrices = {"s1": _symmetric(0.8), "s2": _symmetric(0.2)}
    weights = {"s1": 0.75, "s2": 0.25}
    similarity = raw_convex_fusion(matrices, weights)
    assert similarity[0, 1] == pytest.approx(0.65)
    assert similarity[1, 0] == pytest.approx(0.65)
    assert np.allclose(np.diag(similarity), 1.0)
    dissimilarity = fuse_to_dissimilarity(
        matrices, weights, fusion_mode="raw_convex"
    )
    assert dissimilarity[0, 1] == pytest.approx(0.35)
    assert dissimilarity[1, 0] == pytest.approx(0.35)
    assert np.allclose(np.diag(dissimilarity), 0.0)


def test_variance_calibrated_3x3_matches_hand_fixture() -> None:
    # Upper-triangle S1 = {0.9, 0.3, 0.6}: mean 0.6, var 0.06.
    # Upper-triangle S2 = {0.2, 0.8, 0.5}: mean 0.5, var 0.06.
    # weights 0.7 / 0.3 → off-diag D = {0, 0.4√6, 0.2√6}.
    layer_s1 = np.array(
        [[1.0, 0.9, 0.3], [0.9, 1.0, 0.6], [0.3, 0.6, 1.0]],
        dtype=np.float64,
    )
    layer_s2 = np.array(
        [[1.0, 0.2, 0.8], [0.2, 1.0, 0.5], [0.8, 0.5, 1.0]],
        dtype=np.float64,
    )
    weights = {"s1": 0.7, "s2": 0.3}
    dissimilarity, _calibration = variance_calibrated_fusion(
        {"s1": layer_s1, "s2": layer_s2},
        weights,
    )
    expected = np.array(
        [
            [0.0, 0.0, 0.4 * math.sqrt(6)],
            [0.0, 0.0, 0.2 * math.sqrt(6)],
            [0.4 * math.sqrt(6), 0.2 * math.sqrt(6), 0.0],
        ],
        dtype=np.float64,
    )
    np.testing.assert_allclose(dissimilarity, expected, atol=1e-12, rtol=0)


def test_variance_calibrated_aborts_on_zero_off_diagonal_variance() -> None:
    constant = np.array(
        [[1.0, 0.5, 0.5], [0.5, 1.0, 0.5], [0.5, 0.5, 1.0]],
        dtype=np.float64,
    )
    with pytest.raises(InvalidFusionError, match="off-diagonal variance"):
        variance_calibrated_fusion({"constant": constant}, {"constant": 1.0})


@pytest.mark.parametrize(
    "weights",
    [
        {"s1": 0.5, "s2": 0.4},
        {"s1": 1.2, "s2": -0.2},
        {"s1": 0.75, "unknown": 0.25},
    ],
)
def test_fusion_aborts_on_invalid_weights(weights: dict[str, float]) -> None:
    matrices = {"s1": _symmetric(0.8), "s2": _symmetric(0.2)}
    with pytest.raises(InvalidFusionError):
        fuse_to_dissimilarity(matrices, weights, fusion_mode="raw_convex")


def test_similarity_profile_rejects_non_unit_sum() -> None:
    with pytest.raises(InvalidFusionError, match="sum exactly"):
        SimilarityProfile(name="bad", weights={"a": 0.5, "b": 0.4})


def test_linkage_from_dissimilarity_rejects_ward() -> None:
    dissimilarity = np.array([[0.0, 0.35], [0.35, 0.0]], dtype=np.float64)
    with pytest.raises(InvalidFusionError, match="ward"):
        linkage_from_dissimilarity(dissimilarity, method="ward")
    z_matrix = linkage_from_dissimilarity(dissimilarity, method="average")
    assert z_matrix.shape == (1, 4)


def test_tfidf_and_js_layers_fuse_with_exposed_weights() -> None:
    documents = load_documents()
    taxonomy = StubTaxonomy(
        name="domain",
        label_order=("schedule", "comms", "work", "search"),
        assignments={
            "alpha": {"schedule": 0.7, "comms": 0.1, "work": 0.1, "search": 0.1},
            "bravo": {"schedule": 0.1, "comms": 0.7, "work": 0.1, "search": 0.1},
            "charlie": {"schedule": 0.1, "comms": 0.1, "work": 0.7, "search": 0.1},
            "delta": {"schedule": 0.1, "comms": 0.1, "work": 0.1, "search": 0.7},
        },
    )
    builder = OrchardBuilder(
        taxonomies=[taxonomy],
        include_semantic_with_taxonomies=True,
        fusion_mode="raw_convex",
        semantic_weights={"tfidf_cosine": 0.75, "domain_raw_js": 0.25},
    )
    params = builder.get_params()
    assert params["fusion_mode"] == "raw_convex"
    assert params["profiles"]["semantic"]["weights"] == {
        "tfidf_cosine": 0.75,
        "domain_raw_js": 0.25,
    }
    orchard = builder.build(documents)
    assert "semantic" in orchard.trees
    default = OrchardBuilder(taxonomies=[]).build(documents)
    assert orchard.tree("semantic").linkage.tolist() != default.tree(
        "semantic"
    ).linkage.tolist()


def test_default_semantic_profile_is_inspectable_and_single_layer() -> None:
    builder = OrchardBuilder(taxonomies=[])
    params = builder.get_params()
    assert params["fusion_mode"] == "variance_calibrated"
    assert params["profiles"]["semantic"]["weights"] == {"tfidf_cosine": 1.0}
    assert params["profiles"]["semantic"]["fusion_mode"] == "variance_calibrated"
    assert "fused" not in str(params).lower()


def test_default_taxonomy_trees_match_js_only_path() -> None:
    documents = load_documents()
    orchard = OrchardBuilder().build(documents)
    for taxonomy in default_taxonomies():
        distributions = taxonomy.transform(orchard.documents)
        similarity = jensen_shannon_matrix(distributions)
        expected = linkage_from_similarity(similarity, method="average")
        np.testing.assert_array_equal(orchard.tree(taxonomy.name).linkage, expected)


def test_default_semantic_tree_matches_tfidf_only_path() -> None:
    documents = load_documents()
    orchard = OrchardBuilder(taxonomies=[]).build(documents)
    backend = TfidfEmbeddingBackend()
    features = backend.encode([document.text for document in orchard.documents])
    similarity = cosine_matrix(features, signed=False)
    expected = linkage_from_similarity(similarity, method="average")
    np.testing.assert_array_equal(orchard.tree("semantic").linkage, expected)


def test_layer_registry_validates_tfidf_cosine() -> None:
    documents = load_documents()
    registry = LayerRegistry([TfidfCosineLayer()])
    matrix = registry.compute("tfidf_cosine", documents)
    assert matrix.shape == (len(documents), len(documents))
    assert np.allclose(np.diag(matrix), 1.0)


def test_layer_registry_rejects_unknown_name() -> None:
    registry = LayerRegistry([MatrixLayer(name="s1", similarity=_symmetric(0.8))])
    with pytest.raises(InvalidFusionError, match="unknown layer"):
        registry.compute("missing")
