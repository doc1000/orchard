"""Phase 4: fused no-app taxonomy profiles (fake MiniLM + fake heads)."""

from __future__ import annotations

import numpy as np
import pytest

from orchard import (
    InvalidFusionError,
    MissingOptionalDependencyError,
    OrchardBuilder,
    SimilarityProfile,
    StubTaxonomy,
)
from orchard.backends.minilm import MiniLMEmbeddingBackend
from orchard.backends.modernbert import MODERNBERT_DIMENSIONS, ModernBERTFeatureBackend
from orchard.backends.tfidf import TfidfEmbeddingBackend
from orchard.fixtures import load_documents
from orchard.taxonomy import DomainTaxonomy, FunctionTaxonomy

DOMAIN_BOTH = {
    "domain_raw_js": 0.48,
    "description_minilm_centered_cosine": 0.23,
    "tfidf_cosine": 0.14,
    "function_raw_js": 0.15,
}
FUNCTION_BOTH = {
    "function_raw_js": 0.48,
    "description_minilm_centered_cosine": 0.25,
    "tfidf_cosine": 0.15,
    "domain_raw_js": 0.12,
}
DOMAIN_ONLY = {
    "domain_raw_js": 0.63,
    "description_minilm_centered_cosine": 0.23,
    "tfidf_cosine": 0.14,
}
FUNCTION_ONLY = {
    "function_raw_js": 0.60,
    "description_minilm_centered_cosine": 0.25,
    "tfidf_cosine": 0.15,
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


def _neural_builder(**kwargs: object) -> OrchardBuilder:
    documents = kwargs.pop("documents", None)
    n_docs = len(documents) if documents is not None else len(load_documents())
    kwargs.setdefault("embedding_backend", _fake_minilm(n_docs))
    kwargs.setdefault("taxonomy_classifier_backend", _fake_modernbert())
    return OrchardBuilder(**kwargs)


def test_default_builder_uses_both_taxonomy_fused_dicts() -> None:
    documents = load_documents()
    builder = _neural_builder()
    orchard = builder.build(documents)
    assert orchard.tree_ids == ("domain", "function")
    assert "fused" not in orchard.tree_ids
    assert "mixed" not in orchard.tree_ids
    params = builder.get_params()
    assert params["fusion_mode"] == "variance_calibrated"
    assert params["profiles"]["domain"]["fusion_mode"] == "variance_calibrated"
    assert params["profiles"]["domain"]["weights"] == DOMAIN_BOTH
    assert params["profiles"]["function"]["weights"] == FUNCTION_BOTH
    assert "domain_raw_js" in params["active_layers"]["domain"]
    assert "function_raw_js" in params["active_layers"]["domain"]
    assert "domain_raw_js" in params["active_layers"]["function"]
    assert "function_raw_js" in params["active_layers"]["function"]
    assert params["taxonomy_transform"] == "modernbert_logistic"
    assert "fusion_mode" in params
    assert orchard.metadata["fusion_mode"] == "variance_calibrated"
    assert set(orchard.metadata["layer_checksums"]) >= {
        "domain_raw_js",
        "function_raw_js",
        "tfidf_cosine",
        "description_minilm_centered_cosine",
    }


def test_single_taxonomy_uses_matching_dict_without_renormalize() -> None:
    documents = load_documents()
    encoder = _fake_modernbert()
    domain = DomainTaxonomy.load_default(classifier_backend=encoder)
    builder = OrchardBuilder(
        taxonomies=[domain],
        embedding_backend=_fake_minilm(len(documents)),
    )
    orchard = builder.build(documents)
    params = builder.get_params()
    assert orchard.tree_ids == ("domain",)
    assert params["profiles"]["domain"]["weights"] == DOMAIN_ONLY
    assert "function_raw_js" not in params["profiles"]["domain"]["weights"]
    assert params["taxonomy_transform"] == "modernbert_logistic"

    function = FunctionTaxonomy.load_default(classifier_backend=encoder)
    builder = OrchardBuilder(
        taxonomies=[function],
        embedding_backend=_fake_minilm(len(documents)),
    )
    params = builder.get_params()
    assert params["profiles"]["function"]["weights"] == FUNCTION_ONLY
    assert "domain_raw_js" not in params["profiles"]["function"]["weights"]


def test_profile_and_weight_map_overrides() -> None:
    documents = load_documents()
    builder = _neural_builder(
        taxonomy_weights={"domain": {"domain_raw_js": 1.0}},
        profiles={
            "function": SimilarityProfile(
                name="function",
                weights={"function_raw_js": 0.7, "tfidf_cosine": 0.3},
                fusion_mode="raw_convex",
            )
        },
    )
    orchard = builder.build(documents)
    params = builder.get_params()
    assert params["profiles"]["domain"]["weights"] == {"domain_raw_js": 1.0}
    assert params["profiles"]["function"]["weights"] == {
        "function_raw_js": 0.7,
        "tfidf_cosine": 0.3,
    }
    assert params["profiles"]["function"]["fusion_mode"] == "raw_convex"
    assert orchard.tree_ids == ("domain", "function")


def test_caller_weights_that_do_not_sum_to_one_abort() -> None:
    with pytest.raises(InvalidFusionError, match="sum exactly"):
        _neural_builder(taxonomy_weights={"domain": {"domain_raw_js": 0.5}})


def test_missing_embeddings_with_heads_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("orchard.builder.embeddings_extra_available", lambda: False)
    with pytest.raises(MissingOptionalDependencyError, match="orchard\\[embeddings\\]"):
        OrchardBuilder(taxonomy_classifier_backend=_fake_modernbert())
    with pytest.raises(MissingOptionalDependencyError, match="allow_offline_fallback"):
        OrchardBuilder(taxonomy_classifier_backend=_fake_modernbert())


def test_offline_fallback_stays_single_layer_js(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("orchard.builder.embeddings_extra_available", lambda: False)
    monkeypatch.setattr("orchard.builder.taxonomy_ml_extra_available", lambda: False)
    monkeypatch.setattr("orchard.taxonomy.taxonomy_ml_extra_available", lambda: False)
    builder = OrchardBuilder(allow_offline_fallback=True)
    orchard = builder.build(load_documents())
    params = builder.get_params()
    assert params["offline_fallback"] is True
    assert params["taxonomy_transform"] == "cue"
    assert params["profiles"]["domain"]["weights"] == {"domain_raw_js": 1.0}
    assert params["profiles"]["function"]["weights"] == {"function_raw_js": 1.0}
    assert orchard.tree_ids == ("domain", "function")


def test_explicit_tfidf_backend_does_not_drop_minilm_into_renormalized_profile() -> None:
    builder = OrchardBuilder(
        taxonomies=[
            DomainTaxonomy.load_default(classifier_backend=_fake_modernbert()),
            FunctionTaxonomy.load_default(classifier_backend=_fake_modernbert()),
        ],
        embedding_backend=TfidfEmbeddingBackend(),
    )
    params = builder.get_params()
    assert params["profiles"]["domain"]["weights"] == {"domain_raw_js": 1.0}
    assert params["profiles"]["function"]["weights"] == {"function_raw_js": 1.0}
    assert params["embedding_backend"] == "TfidfEmbeddingBackend"


def test_shared_layers_computed_once() -> None:
    documents = load_documents()
    encodes: list[int] = []
    rng = np.random.default_rng(20260725)
    vectors = rng.normal(size=(len(documents), 8))

    def encode(_texts):
        encodes.append(1)
        return vectors

    domain_calls = {"n": 0}
    function_calls = {"n": 0}

    class CountingDomain(StubTaxonomy):
        def transform(self, documents):  # type: ignore[no-untyped-def]
            domain_calls["n"] += 1
            return super().transform(documents)

    class CountingFunction(StubTaxonomy):
        def transform(self, documents):  # type: ignore[no-untyped-def]
            function_calls["n"] += 1
            return super().transform(documents)

    domain = CountingDomain(
        name="domain",
        label_order=("schedule", "comms", "work", "search"),
        assignments={
            "alpha": {"schedule": 0.7, "comms": 0.1, "work": 0.1, "search": 0.1},
            "bravo": {"schedule": 0.1, "comms": 0.7, "work": 0.1, "search": 0.1},
            "charlie": {"schedule": 0.1, "comms": 0.1, "work": 0.7, "search": 0.1},
            "delta": {"schedule": 0.1, "comms": 0.1, "work": 0.1, "search": 0.7},
        },
    )
    function = CountingFunction(
        name="function",
        label_order=("create", "send", "find"),
        assignments={
            "alpha": {"create": 0.6, "send": 0.2, "find": 0.2},
            "bravo": {"create": 0.2, "send": 0.6, "find": 0.2},
            "charlie": {"create": 0.6, "send": 0.2, "find": 0.2},
            "delta": {"create": 0.2, "send": 0.2, "find": 0.6},
        },
    )
    builder = OrchardBuilder(
        taxonomies=[domain, function],
        embedding_backend=MiniLMEmbeddingBackend(encode_fn=encode),
        fusion_mode="raw_convex",
    )
    builder.build(documents)
    assert builder.get_params()["profiles"]["domain"]["weights"] == DOMAIN_BOTH
    assert encodes == [1]
    assert domain_calls["n"] == 1
    assert function_calls["n"] == 1


def test_similarity_profile_is_public() -> None:
    profile = SimilarityProfile(
        name="domain",
        weights=DOMAIN_BOTH,
    )
    assert profile.fusion_mode == "variance_calibrated"
    assert profile.to_dict()["weights"] == DOMAIN_BOTH
