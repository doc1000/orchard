"""Phase 3: default Domain + Function taxonomies and replaceability."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from orchard import DomainTaxonomy, FunctionTaxonomy, OrchardBuilder, TaxonomyModel
from orchard.backends.minilm import MiniLMEmbeddingBackend
from orchard.backends.modernbert import MODERNBERT_DIMENSIONS, ModernBERTFeatureBackend
from orchard.fixtures import load_documents


def _force_taxonomy_offline(monkeypatch: pytest.MonkeyPatch) -> None:
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


def _fake_minilm(n_docs: int) -> MiniLMEmbeddingBackend:
    rng = np.random.default_rng(20260725)
    vectors = rng.normal(size=(n_docs, 8))
    return MiniLMEmbeddingBackend(encode_fn=lambda _texts: vectors)


def _fake_modernbert() -> ModernBERTFeatureBackend:
    def encode(texts):
        rng = np.random.default_rng(20260725)
        return rng.normal(size=(len(texts), MODERNBERT_DIMENSIONS))

    return ModernBERTFeatureBackend(encode_fn=encode)


def test_default_domain_function_build(monkeypatch: pytest.MonkeyPatch) -> None:
    _force_taxonomy_offline(monkeypatch)
    builder = OrchardBuilder(allow_offline_fallback=True)
    orchard = builder.build(load_documents())
    assert orchard.tree_ids == ("domain", "function")
    assert "semantic" not in orchard.trees
    assert orchard.tree("domain").leaf_count == 4
    assert orchard.tree("function").leaf_count == 4
    params = builder.get_params()
    assert params["taxonomy_transform"] == "cue"
    assert params["profiles"]["domain"]["weights"] == {"domain_raw_js": 1.0}
    assert params["profiles"]["function"]["weights"] == {"function_raw_js": 1.0}


def test_default_domain_function_fused_neural_profiles() -> None:
    documents = load_documents()
    builder = OrchardBuilder(
        embedding_backend=_fake_minilm(len(documents)),
        taxonomy_classifier_backend=_fake_modernbert(),
    )
    orchard = builder.build(documents)
    assert orchard.tree_ids == ("domain", "function")
    assert "fused" not in orchard.tree_ids
    assert "mixed" not in orchard.tree_ids
    params = builder.get_params()
    assert params["fusion_mode"] == "variance_calibrated"
    assert params["taxonomy_transform"] == "modernbert_logistic"
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


def test_explicit_default_loaders_match_builder(monkeypatch: pytest.MonkeyPatch) -> None:
    _force_taxonomy_offline(monkeypatch)
    domain = DomainTaxonomy.load_default(allow_offline_fallback=True)
    function = FunctionTaxonomy.load_default(allow_offline_fallback=True)
    assert domain.name == "domain"
    assert function.name == "function"
    assert domain.provenance
    assert "AppWorld" not in domain.provenance or "no AppWorld" in domain.provenance
    assert domain.taxonomy_transform == "cue"

    orchard = OrchardBuilder(taxonomies=[domain, function]).build(load_documents())
    assert set(orchard.trees) == {"domain", "function"}


def test_taxonomy_artifacts_are_replaceable(tmp_path: Path) -> None:
    domain = TaxonomyModel.load_default("domain")
    path = domain.save(tmp_path / "domain.json")
    reloaded = TaxonomyModel.load(path)
    assert reloaded.label_order == domain.label_order
    assert reloaded.labels["domain.work"].cues

    # User can load a modified definition and build with it.
    payload = reloaded.to_definition()
    payload["labels"][0]["cues"] = ["customcue"]
    custom_path = tmp_path / "custom_domain.json"
    custom_path.write_text(
        __import__("json").dumps(payload),
        encoding="utf-8",
    )
    custom = TaxonomyModel.load(custom_path)
    orchard = OrchardBuilder(
        taxonomies=[custom, TaxonomyModel.load_default("function")]
    ).build(load_documents())
    assert "domain" in orchard.trees
