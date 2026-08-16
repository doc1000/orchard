"""Packaged similarity-profile weight dicts."""

from __future__ import annotations

import json
from collections.abc import Sequence
from importlib import resources
from typing import Any

PACKAGED_MINILM_LAYER = "description_minilm_centered_cosine"
TAXONOMY_PROFILE_NAMES = frozenset({"domain", "function"})


def load_profile_payload(name: str) -> dict[str, Any]:
    """Load a packaged profile JSON payload by tree id."""
    resource = resources.files("orchard.assets.profiles").joinpath(f"{name}.json")
    return json.loads(resource.read_text(encoding="utf-8"))


def load_semantic_profile() -> dict[str, Any]:
    """Load the packaged no-app semantic profile (0.66 MiniLM / 0.34 TF-IDF)."""
    return load_profile_payload("semantic")


def _rename_minilm_layer(weights: dict[str, float], minilm_layer: str) -> dict[str, float]:
    if minilm_layer != PACKAGED_MINILM_LAYER and PACKAGED_MINILM_LAYER in weights:
        weights[minilm_layer] = weights.pop(PACKAGED_MINILM_LAYER)
    return {name: float(value) for name, value in weights.items()}


def load_semantic_weights(*, minilm_layer: str) -> dict[str, float]:
    """Return semantic weights with the MiniLM layer name for the active transform."""
    payload = load_semantic_profile()
    return _rename_minilm_layer(dict(payload["weights"]), minilm_layer)


def load_taxonomy_weights(
    tree_id: str,
    *,
    taxonomy_names: Sequence[str],
    minilm_layer: str,
    with_app: bool = False,
) -> dict[str, float]:
    """Return the packaged taxonomy dict for the present layer subset.

    Both Domain and Function present → cross-taxonomy JS dict.
    Only one of those taxonomies → matching single-taxonomy dict (missing JS
    layer is absent; never renormalized).
    ``with_app=True`` selects the G3 ``app_exact_match=0.03`` dicts.
    """
    if tree_id not in TAXONOMY_PROFILE_NAMES:
        raise ValueError(f"no packaged taxonomy profile for {tree_id!r}")
    payload = load_profile_payload(tree_id)
    present = set(taxonomy_names)
    other = "function" if tree_id == "domain" else "domain"
    if with_app:
        key = "with_app_weights" if other in present else "with_app_single_taxonomy_weights"
    elif other in present:
        key = "weights"
    else:
        key = "single_taxonomy_weights"
    return _rename_minilm_layer(dict(payload[key]), minilm_layer)
