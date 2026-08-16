"""Phase 5: optional app_exact_match layer and with-app profile dicts."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from orchard import (
    InvalidFusionError,
    Orchard,
    OrchardBuilder,
    SimilarityProfile,
)
from orchard.backends.family import FAMILY_LAYER_NAME, AppExactMatchLayer
from orchard.backends.minilm import MiniLMEmbeddingBackend
from orchard.backends.modernbert import MODERNBERT_DIMENSIONS, ModernBERTFeatureBackend
from orchard.document import Document
from orchard.fixtures import load_documents
from orchard.taxonomy import DomainTaxonomy, FunctionTaxonomy

DOMAIN_BOTH_NO_APP = {
    "domain_raw_js": 0.48,
    "description_minilm_centered_cosine": 0.23,
    "tfidf_cosine": 0.14,
    "function_raw_js": 0.15,
}
FUNCTION_BOTH_NO_APP = {
    "function_raw_js": 0.48,
    "description_minilm_centered_cosine": 0.25,
    "tfidf_cosine": 0.15,
    "domain_raw_js": 0.12,
}
DOMAIN_BOTH_WITH_APP = {
    "app_exact_match": 0.03,
    "description_minilm_centered_cosine": 0.23,
    "domain_raw_js": 0.45,
    "function_raw_js": 0.15,
    "tfidf_cosine": 0.14,
}
FUNCTION_BOTH_WITH_APP = {
    "app_exact_match": 0.03,
    "description_minilm_centered_cosine": 0.25,
    "domain_raw_js": 0.12,
    "function_raw_js": 0.45,
    "tfidf_cosine": 0.15,
}
DOMAIN_ONLY_WITH_APP = {
    "app_exact_match": 0.03,
    "description_minilm_centered_cosine": 0.23,
    "domain_raw_js": 0.60,
    "tfidf_cosine": 0.14,
}
FUNCTION_ONLY_WITH_APP = {
    "app_exact_match": 0.03,
    "description_minilm_centered_cosine": 0.25,
    "function_raw_js": 0.57,
    "tfidf_cosine": 0.15,
}
SEMANTIC_WEIGHTS = {
    "description_minilm_centered_cosine": 0.66,
    "tfidf_cosine": 0.34,
}
FAMILY_BY_ID = {
    "alpha": "mail",
    "bravo": "mail",
    "charlie": "calendar",
    "delta": "calendar",
}


def _fake_minilm(n_docs: int) -> MiniLMEmbeddingBackend:
    rng = np.random.default_rng(20260725)
    vectors = rng.normal(size=(n_docs, 8))
    return MiniLMEmbeddingBackend(encode_fn=lambda _texts: vectors)


def _fake_modernbert() -> ModernBERTFeatureBackend:
    def encode(texts):
        rng = np.random.default_rng(20260725)
        return rng.normal(size=(len(texts), MODERNBERT_DIMENSIONS))

    return ModernBERTFeatureBackend(encode_fn=encode)


def _with_family(
    documents: list[Document],
    *,
    key: str = "app_name",
    values: dict[str, str] | None = None,
) -> list[Document]:
    mapping = FAMILY_BY_ID if values is None else values
    updated: list[Document] = []
    for document in documents:
        metadata = dict(document.metadata)
        if document.item_id in mapping:
            metadata[key] = mapping[document.item_id]
        updated.append(replace(document, metadata=metadata))
    return updated


def _neural_builder(**kwargs: object) -> OrchardBuilder:
    documents = kwargs.pop("documents", None)
    n_docs = len(documents) if documents is not None else len(load_documents())
    kwargs.setdefault("embedding_backend", _fake_minilm(n_docs))
    kwargs.setdefault("taxonomy_classifier_backend", _fake_modernbert())
    return OrchardBuilder(**kwargs)


def test_app_exact_match_layer_same_family_is_one() -> None:
    documents = _with_family(load_documents())
    matrix = AppExactMatchLayer(metadata_key="app_name").matrix(documents)
    assert matrix.shape == (4, 4)
    assert np.allclose(np.diag(matrix), 1.0)
    assert matrix[0, 1] == 1.0
    assert matrix[1, 0] == 1.0
    assert matrix[2, 3] == 1.0
    assert matrix[0, 2] == 0.0
    assert matrix[0, 3] == 0.0
    assert np.isfinite(matrix).all()
    np.testing.assert_allclose(matrix, matrix.T)


def test_family_key_and_complete_metadata_uses_with_app_dicts() -> None:
    documents = _with_family(load_documents())
    builder = _neural_builder(family_metadata_key="app_name")
    orchard = builder.build(documents)
    params = builder.get_params()
    assert orchard.tree_ids == ("domain", "function")
    assert "fused" not in orchard.tree_ids
    assert "mixed" not in orchard.tree_ids
    assert params["family_metadata_key"] == "app_name"
    assert params["app_exact_match_active"] is True
    assert params["profiles"]["domain"]["weights"] == DOMAIN_BOTH_WITH_APP
    assert params["profiles"]["function"]["weights"] == FUNCTION_BOTH_WITH_APP
    assert FAMILY_LAYER_NAME in params["active_layers"]["domain"]
    assert FAMILY_LAYER_NAME in params["active_layers"]["function"]
    assert "domain_raw_js" in params["active_layers"]["domain"]
    assert "function_raw_js" in params["active_layers"]["domain"]
    assert orchard.metadata["app_exact_match_active"] is True
    assert FAMILY_LAYER_NAME in orchard.metadata["layer_checksums"]
    family = orchard.layer_matrices[FAMILY_LAYER_NAME]
    assert family[0, 1] == 1.0
    assert family[2, 3] == 1.0
    assert family[0, 2] == 0.0


def test_both_taxonomies_include_app_and_both_js_layers() -> None:
    documents = _with_family(load_documents())
    params = _neural_builder(family_metadata_key="app_name").build(documents)
    builder_params = params.metadata["builder"]
    for tree_id in ("domain", "function"):
        layers = builder_params["active_layers"][tree_id]
        assert FAMILY_LAYER_NAME in layers
        assert "domain_raw_js" in layers
        assert "function_raw_js" in layers


def test_single_taxonomy_with_app_uses_matching_dict() -> None:
    documents = _with_family(load_documents())
    encoder = _fake_modernbert()
    domain = DomainTaxonomy.load_default(classifier_backend=encoder)
    builder = OrchardBuilder(
        taxonomies=[domain],
        embedding_backend=_fake_minilm(len(documents)),
        family_metadata_key="app_name",
    )
    orchard = builder.build(documents)
    params = builder.get_params()
    assert orchard.tree_ids == ("domain",)
    assert params["profiles"]["domain"]["weights"] == DOMAIN_ONLY_WITH_APP
    assert "function_raw_js" not in params["profiles"]["domain"]["weights"]
    assert sum(params["profiles"]["domain"]["weights"].values()) == pytest.approx(1.0)

    function = FunctionTaxonomy.load_default(classifier_backend=encoder)
    builder = OrchardBuilder(
        taxonomies=[function],
        embedding_backend=_fake_minilm(len(documents)),
        family_metadata_key="family",
    )
    orchard = builder.build(_with_family(documents, key="family"))
    params = builder.get_params()
    assert params["profiles"]["function"]["weights"] == FUNCTION_ONLY_WITH_APP
    assert "domain_raw_js" not in params["profiles"]["function"]["weights"]
    assert sum(params["profiles"]["function"]["weights"].values()) == pytest.approx(1.0)


def test_partial_family_metadata_is_loud_error_and_keeps_no_app_weights() -> None:
    documents = load_documents()
    partial = _with_family(
        documents,
        values={"alpha": "mail", "bravo": "mail", "charlie": "calendar"},
    )
    builder = _neural_builder(family_metadata_key="app_name")
    assert builder.get_params()["profiles"]["domain"]["weights"] == DOMAIN_BOTH_NO_APP
    with pytest.raises(InvalidFusionError, match="delta"):
        builder.build(partial)
    assert builder.get_params()["profiles"]["domain"]["weights"] == DOMAIN_BOTH_NO_APP
    assert builder.get_params()["app_exact_match_active"] is False

    empty = _with_family(documents)
    empty[1] = replace(empty[1], metadata={**empty[1].metadata, "app_name": "  "})
    builder = _neural_builder(family_metadata_key="app_name")
    with pytest.raises(InvalidFusionError, match="bravo"):
        builder.build(empty)
    assert builder.get_params()["profiles"]["function"]["weights"] == FUNCTION_BOTH_NO_APP


def test_unset_key_stays_on_phase4_no_app_dicts() -> None:
    documents = _with_family(load_documents())
    builder = _neural_builder()
    orchard = builder.build(documents)
    params = builder.get_params()
    assert params["family_metadata_key"] is None
    assert params["app_exact_match_active"] is False
    assert params["profiles"]["domain"]["weights"] == DOMAIN_BOTH_NO_APP
    assert params["profiles"]["function"]["weights"] == FUNCTION_BOTH_NO_APP
    assert FAMILY_LAYER_NAME not in orchard.metadata["layer_checksums"]
    assert "fused" not in orchard.tree_ids
    assert "mixed" not in orchard.tree_ids


def test_profile_names_family_without_key_still_requires_metadata() -> None:
    documents = load_documents()
    builder = _neural_builder(
        profiles={
            "domain": SimilarityProfile(
                name="domain",
                weights=DOMAIN_BOTH_WITH_APP,
            )
        }
    )
    with pytest.raises(InvalidFusionError, match="app_exact_match"):
        builder.build(documents)
    orchard = builder.build(_with_family(documents))
    params = builder.get_params()
    assert params["family_metadata_key"] is None
    assert params["app_exact_match_active"] is True
    assert params["profiles"]["domain"]["weights"] == DOMAIN_BOTH_WITH_APP
    assert params["profiles"]["function"]["weights"] == FUNCTION_BOTH_NO_APP
    assert FAMILY_LAYER_NAME in orchard.layer_matrices


def test_semantic_stays_no_app_when_family_is_on() -> None:
    documents = _with_family(load_documents())
    builder = _neural_builder(
        family_metadata_key="app_name",
        include_semantic_with_taxonomies=True,
    )
    orchard = builder.build(documents)
    params = builder.get_params()
    assert params["profiles"]["semantic"]["weights"] == SEMANTIC_WEIGHTS
    assert FAMILY_LAYER_NAME not in params["profiles"]["semantic"]["weights"]
    assert params["profiles"]["domain"]["weights"] == DOMAIN_BOTH_WITH_APP
    assert orchard.tree_ids == ("domain", "function", "semantic")
    assert "fused" not in orchard.tree_ids


def test_offline_fallback_does_not_auto_enable_family(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("orchard.builder.embeddings_extra_available", lambda: False)
    monkeypatch.setattr("orchard.builder.taxonomy_ml_extra_available", lambda: False)
    monkeypatch.setattr("orchard.taxonomy.taxonomy_ml_extra_available", lambda: False)
    documents = _with_family(load_documents())
    builder = OrchardBuilder(
        allow_offline_fallback=True,
        family_metadata_key="app_name",
    )
    orchard = builder.build(documents)
    params = builder.get_params()
    assert params["offline_fallback"] is True
    assert params["app_exact_match_active"] is False
    assert params["profiles"]["domain"]["weights"] == {"domain_raw_js": 1.0}
    assert params["profiles"]["function"]["weights"] == {"function_raw_js": 1.0}
    assert orchard.tree_ids == ("domain", "function")


def test_all_same_app_aborts_variance_calibrated() -> None:
    documents = _with_family(
        load_documents(),
        values={item_id: "mail" for item_id in FAMILY_BY_ID},
    )
    builder = _neural_builder(family_metadata_key="app_name")
    with pytest.raises(InvalidFusionError, match="app_exact_match"):
        builder.build(documents)
    assert builder.get_params()["profiles"]["domain"]["weights"] == DOMAIN_BOTH_WITH_APP


def test_layer_npz_persists_only_when_policy_allows(tmp_path) -> None:
    documents = _with_family(load_documents())
    default = _neural_builder(family_metadata_key="app_name").build(documents)
    out = default.save(tmp_path / "never")
    assert not (out / "layer_matrices.npz").is_file()
    loaded = Orchard.load(out)
    assert loaded.layer_matrices == {}
    assert loaded.metadata["app_exact_match_active"] is True

    always = _neural_builder(
        family_metadata_key="app_name",
        layer_matrix_persist="always",
    ).build(documents)
    out = always.save(tmp_path / "always")
    assert (out / "layer_matrices.npz").is_file()
    loaded = Orchard.load(out)
    np.testing.assert_allclose(
        loaded.layer_matrices[FAMILY_LAYER_NAME],
        always.layer_matrices[FAMILY_LAYER_NAME],
    )
