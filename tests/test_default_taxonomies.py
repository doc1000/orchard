"""Phase 3: default Domain + Function taxonomies and replaceability."""

from __future__ import annotations

from pathlib import Path

from orchard import DomainTaxonomy, FunctionTaxonomy, OrchardBuilder, TaxonomyModel
from orchard.fixtures import load_documents


def test_default_domain_function_build() -> None:
    orchard = OrchardBuilder().build(load_documents())
    assert orchard.tree_ids == ("domain", "function")
    assert "semantic" not in orchard.trees
    assert orchard.tree("domain").leaf_count == 4
    assert orchard.tree("function").leaf_count == 4


def test_explicit_default_loaders_match_builder() -> None:
    domain = DomainTaxonomy.load_default()
    function = FunctionTaxonomy.load_default()
    assert domain.name == "domain"
    assert function.name == "function"
    assert domain.provenance
    assert "AppWorld" not in domain.provenance or "no AppWorld" in domain.provenance

    orchard = OrchardBuilder(taxonomies=[domain, function]).build(load_documents())
    assert set(orchard.trees) == {"domain", "function"}


def test_taxonomy_artifacts_are_replaceable(tmp_path: Path) -> None:
    domain = DomainTaxonomy.load_default()
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
    orchard = OrchardBuilder(taxonomies=[custom, FunctionTaxonomy.load_default()]).build(
        load_documents()
    )
    assert "domain" in orchard.trees
