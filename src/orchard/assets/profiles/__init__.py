"""Packaged similarity-profile weight dicts."""

from __future__ import annotations

import json
from importlib import resources
from typing import Any


def load_semantic_profile() -> dict[str, Any]:
    """Load the packaged no-app semantic profile (0.66 MiniLM / 0.34 TF-IDF)."""
    resource = resources.files("orchard.assets.profiles").joinpath("semantic.json")
    return json.loads(resource.read_text(encoding="utf-8"))


def load_semantic_weights(*, minilm_layer: str) -> dict[str, float]:
    """Return semantic weights with the MiniLM layer name for the active transform."""
    payload = load_semantic_profile()
    weights = dict(payload["weights"])
    packaged = "description_minilm_centered_cosine"
    if minilm_layer != packaged and packaged in weights:
        weights[minilm_layer] = weights.pop(packaged)
    return {name: float(value) for name, value in weights.items()}
