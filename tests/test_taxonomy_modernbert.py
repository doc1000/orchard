"""Phase 3 ModernBERT taxonomy heads (CI: fake encoder, no download)."""

from __future__ import annotations

import numpy as np
import pytest

from orchard import MissingOptionalDependencyError, OrchardBuilder, TaxonomyModel
from orchard.backends.minilm import MiniLMEmbeddingBackend
from orchard.backends.modernbert import (
    FEATURE_CONFIG,
    MODERNBERT_DIMENSIONS,
    MODERNBERT_MODEL_ID,
    MODERNBERT_REVISION,
    ModernBERTFeatureBackend,
    modernbert_cache_key,
    modernbert_snapshot_available,
)
from orchard.backends.similarity import (
    jensen_shannon_matrix,
    linkage_from_similarity,
    validate_similarity_matrix,
)
from orchard.backends.taxonomy_heads import (
    SOURCE_STUDENT_SHA256,
    load_packaged_arrays,
    load_sidecar,
    remap_head_to_label_order,
)
from orchard.document import Document
from orchard.fixtures import load_documents
from orchard.taxonomy import DomainTaxonomy, FunctionTaxonomy, default_taxonomies


def _fake_modernbert(n_docs: int | None = None) -> ModernBERTFeatureBackend:
    def encode(texts):
        count = n_docs if n_docs is not None else len(texts)
        rng = np.random.default_rng(20260725)
        return rng.normal(size=(count, MODERNBERT_DIMENSIONS))

    return ModernBERTFeatureBackend(encode_fn=encode)


def _fake_minilm(n_docs: int) -> MiniLMEmbeddingBackend:
    rng = np.random.default_rng(20260725)
    vectors = rng.normal(size=(n_docs, 8))
    return MiniLMEmbeddingBackend(encode_fn=lambda _texts: vectors)


def test_packaged_heads_match_label_order_and_source_shas() -> None:
    for name in ("domain", "function"):
        taxonomy = TaxonomyModel.load_default(name)
        coef, intercept, classes = load_packaged_arrays(name)
        sidecar = load_sidecar(name)
        stored = [str(label) for label in classes.tolist()]
        assert set(stored) == set(taxonomy.label_order)
        assert coef.shape == (len(taxonomy.label_order), 768)
        assert intercept.shape == (len(taxonomy.label_order),)
        assert sidecar["source_student_sha256"] == SOURCE_STUDENT_SHA256[name]
        assert sidecar["feature_model"]["id"] == MODERNBERT_MODEL_ID
        assert sidecar["feature_model"]["revision"] == MODERNBERT_REVISION
        remapped_coef, remapped_intercept, remapped_classes = remap_head_to_label_order(
            coef, intercept, stored, taxonomy.label_order
        )
        assert remapped_classes.astype(str).tolist() == list(taxonomy.label_order)
        assert remapped_coef.shape == coef.shape
        assert remapped_intercept.shape == intercept.shape
        if stored != list(taxonomy.label_order):
            assert not np.array_equal(remapped_coef, coef)


def test_fake_encoder_packaged_head_transform_rows() -> None:
    documents = load_documents()
    encoder = _fake_modernbert()
    domain = DomainTaxonomy.load_default(classifier_backend=encoder)
    rows = domain.transform(documents)
    assert rows.shape == (len(documents), len(domain.label_order))
    np.testing.assert_allclose(rows.sum(axis=1), 1.0, atol=1e-12)
    assert domain.classifier is not None
    assert domain.classifier.classes_.tolist() == list(domain.label_order)
    assert domain.taxonomy_transform == "modernbert_logistic"
    similarity = jensen_shannon_matrix(rows)
    validate_similarity_matrix(similarity, [doc.item_id for doc in documents])


def test_taxonomy_ml_extra_loads_packaged_heads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "orchard.builder.taxonomy_ml_extra_available",
        lambda: True,
    )
    monkeypatch.setattr(
        "orchard.taxonomy.taxonomy_ml_extra_available",
        lambda: True,
    )
    builder = OrchardBuilder(embedding_backend=_fake_minilm(4))
    assert isinstance(builder.taxonomy_classifier_backend, ModernBERTFeatureBackend)
    params = builder.get_params()
    assert params["taxonomy_transform"] == "modernbert_logistic"
    assert params["taxonomy_model_id"] == MODERNBERT_MODEL_ID
    assert params["taxonomy_model_revision"] == MODERNBERT_REVISION
    assert params["profiles"]["domain"]["weights"]["domain_raw_js"] == 0.48
    assert params["profiles"]["function"]["weights"]["function_raw_js"] == 0.48
    assert {taxonomy.taxonomy_transform for taxonomy in builder.taxonomies} == {
        "modernbert_logistic"
    }


def test_injected_backend_records_modernbert_logistic_not_cue() -> None:
    documents = load_documents()
    encoder = _fake_modernbert()
    builder = OrchardBuilder(
        taxonomy_classifier_backend=encoder,
        embedding_backend=_fake_minilm(len(documents)),
    )
    orchard = builder.build(documents)
    assert orchard.tree_ids == ("domain", "function")
    params = builder.get_params()
    assert params["taxonomy_transform"] == "modernbert_logistic"
    assert params["taxonomy_classifier_backend"] == "ModernBERTFeatureBackend"
    assert params["taxonomy_model_id"] == MODERNBERT_MODEL_ID
    assert params["taxonomy_model_revision"] == MODERNBERT_REVISION
    assert params["taxonomy_pooling"] == FEATURE_CONFIG["pooling"]
    assert params["fusion_mode"] == "variance_calibrated"
    assert params["profiles"]["domain"]["weights"] == {
        "domain_raw_js": 0.48,
        "description_minilm_centered_cosine": 0.23,
        "tfidf_cosine": 0.14,
        "function_raw_js": 0.15,
    }
    assert params["profiles"]["function"]["weights"] == {
        "function_raw_js": 0.48,
        "description_minilm_centered_cosine": 0.25,
        "tfidf_cosine": 0.15,
        "domain_raw_js": 0.12,
    }
    assert "cue" != params["taxonomy_transform"]


def test_missing_extra_default_builder_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "orchard.builder.taxonomy_ml_extra_available",
        lambda: False,
    )
    monkeypatch.setattr(
        "orchard.taxonomy.taxonomy_ml_extra_available",
        lambda: False,
    )
    with pytest.raises(MissingOptionalDependencyError, match="orchard\\[taxonomy-ml\\]"):
        OrchardBuilder()
    with pytest.raises(MissingOptionalDependencyError, match="allow_offline_fallback"):
        default_taxonomies()


def test_offline_fallback_uses_cue_and_records_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    documents = load_documents()
    builder = OrchardBuilder(allow_offline_fallback=True)
    orchard = builder.build(documents)
    assert orchard.tree_ids == ("domain", "function")
    params = builder.get_params()
    assert params["taxonomy_transform"] == "cue"
    assert params["taxonomy_model_id"] is None
    assert params["taxonomy_model_revision"] is None
    assert params["allow_offline_fallback"] is True
    assert params["offline_fallback"] is True
    assert params["profiles"]["domain"]["weights"] == {"domain_raw_js": 1.0}
    assert params["profiles"]["function"]["weights"] == {"function_raw_js": 1.0}
    cue_domain = TaxonomyModel.load_default("domain")
    cue_function = TaxonomyModel.load_default("function")
    for taxonomy in (cue_domain, cue_function):
        distributions = taxonomy.transform(orchard.documents)
        similarity = jensen_shannon_matrix(distributions)
        expected = linkage_from_similarity(similarity, method="average")
        np.testing.assert_array_equal(orchard.tree(taxonomy.name).linkage, expected)


def test_stub_and_custom_taxonomies_do_not_require_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "orchard.builder.taxonomy_ml_extra_available",
        lambda: False,
    )
    monkeypatch.setattr(
        "orchard.taxonomy.taxonomy_ml_extra_available",
        lambda: False,
    )
    custom = TaxonomyModel.load_default("domain")
    orchard = OrchardBuilder(taxonomies=[custom]).build(load_documents())
    assert orchard.tree_ids == ("domain",)
    assert custom.taxonomy_transform == "cue"


def test_modernbert_cache_key_includes_revision() -> None:
    texts = ["alpha", "bravo"]
    key_a = modernbert_cache_key(texts)
    key_b = modernbert_cache_key(texts, revision="other-revision")
    key_c = modernbert_cache_key(["alpha", "charlie"])
    key_d = modernbert_cache_key(
        texts,
        configuration={**FEATURE_CONFIG, "max_length": 256},
    )
    assert key_a != key_b
    assert key_a != key_c
    assert key_a != key_d
    assert key_a == modernbert_cache_key(texts, model_id=MODERNBERT_MODEL_ID)


def test_load_and_save_head_roundtrip(tmp_path) -> None:
    encoder = _fake_modernbert()
    domain = DomainTaxonomy.load_default(classifier_backend=encoder)
    path = domain.save_head(tmp_path / "domain_head.npz")
    reloaded = TaxonomyModel.load_default("domain").load_head(path, encoder=encoder)
    documents = [
        Document(item_id="a", title="Calendar", text="schedule a reminder"),
        Document(item_id="b", title="Mail", text="send an email"),
    ]
    np.testing.assert_allclose(domain.transform(documents), reloaded.transform(documents))
    assert reloaded.taxonomy_transform == "modernbert_logistic"


def test_function_head_column_order_is_orchard_label_order() -> None:
    encoder = _fake_modernbert()
    function = FunctionTaxonomy.load_default(classifier_backend=encoder)
    rows = function.transform(load_documents())
    assert function.classifier is not None
    assert function.classifier.classes_.tolist() == list(function.label_order)
    assert rows.shape[1] == len(function.label_order)


@pytest.mark.optional_model
def test_live_modernbert_encode_optional(tmp_path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    if not modernbert_snapshot_available():
        pytest.skip("ModernBERT revision snapshot is not present locally")
    backend = ModernBERTFeatureBackend(cache_dir=tmp_path)
    features = backend.encode(["hello world", "another sentence"])
    assert features.shape == (2, MODERNBERT_DIMENSIONS)
    assert np.isfinite(features).all()
