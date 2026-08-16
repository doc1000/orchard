"""Independent similarity layers (D-002, D-027).

Each layer returns an [n, n] similarity matrix: finite, symmetric, [0, 1],
unit diagonal. Layers are never concatenated as feature vectors.

Source contract: tool-tree-demo Phase 3 independent matrices, then fusion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Protocol, Sequence, runtime_checkable

import numpy as np

from orchard.backends.similarity import (
    cosine_matrix,
    jensen_shannon_matrix,
    validate_similarity_matrix,
)
from orchard.backends.tfidf import TfidfEmbeddingBackend
from orchard.document import Document
from orchard.exceptions import InvalidFusionError
from orchard.taxonomy import Taxonomy


@runtime_checkable
class SimilarityLayer(Protocol):
    """Named producer of a finalized pairwise similarity matrix."""

    name: str

    def matrix(self, documents: Sequence[Document], **ctx: Any) -> np.ndarray:
        """Return [n, n] similarity in [0, 1], symmetric, unit diagonal."""
        ...


@dataclass
class TfidfCosineLayer:
    """Document-text TF-IDF cosine (layer name ``tfidf_cosine``)."""

    backend: Any = None
    signed: bool = False
    name: str = "tfidf_cosine"

    def __post_init__(self) -> None:
        if self.backend is None:
            self.backend = TfidfEmbeddingBackend()

    def matrix(self, documents: Sequence[Document], **ctx: Any) -> np.ndarray:
        texts = [document.text for document in documents]
        features = np.asarray(self.backend.encode(texts), dtype=np.float64)
        if features.shape[0] != len(documents):
            raise InvalidFusionError("tfidf_cosine row count must match documents")
        return cosine_matrix(features, signed=self.signed)


@dataclass
class TaxonomyJsLayer:
    """Jensen–Shannon similarity over taxonomy probability rows.

    Layer name defaults to ``{taxonomy.name}_raw_js`` (AppWorld ``*_raw_js``).
    """

    taxonomy: Taxonomy
    name: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            self.name = f"{self.taxonomy.name}_raw_js"

    def matrix(self, documents: Sequence[Document], **ctx: Any) -> np.ndarray:
        distributions = np.asarray(self.taxonomy.transform(documents), dtype=np.float64)
        if distributions.shape[0] != len(documents):
            raise InvalidFusionError(
                f"{self.name} row count must match documents"
            )
        return jensen_shannon_matrix(distributions)


@dataclass
class TaxonomyCosineLayer:
    """Cosine over taxonomy probability rows (``{taxonomy.name}_cosine``)."""

    taxonomy: Taxonomy
    name: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            self.name = f"{self.taxonomy.name}_cosine"

    def matrix(self, documents: Sequence[Document], **ctx: Any) -> np.ndarray:
        distributions = np.asarray(self.taxonomy.transform(documents), dtype=np.float64)
        if distributions.shape[0] != len(documents):
            raise InvalidFusionError(
                f"{self.name} row count must match documents"
            )
        return cosine_matrix(distributions, signed=False)


@dataclass
class MatrixLayer:
    """Inject a precomputed [n, n] similarity (tests and custom layers)."""

    name: str
    similarity: np.ndarray

    def matrix(self, documents: Sequence[Document] = (), **ctx: Any) -> np.ndarray:
        return np.asarray(self.similarity, dtype=np.float64)


class LayerRegistry:
    """Name → layer lookup; computes and validates similarity matrices."""

    def __init__(self, layers: Iterable[SimilarityLayer] | None = None) -> None:
        self._layers: dict[str, SimilarityLayer] = {}
        if layers is not None:
            for layer in layers:
                self.register(layer)

    def register(self, layer: SimilarityLayer) -> None:
        if not getattr(layer, "name", None):
            raise InvalidFusionError("similarity layer must have a non-empty name")
        if layer.name in self._layers:
            raise InvalidFusionError(f"duplicate layer name: {layer.name}")
        self._layers[layer.name] = layer

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._layers)

    def compute(
        self,
        name: str,
        documents: Sequence[Document] = (),
        *,
        item_ids: Sequence[str] | None = None,
        **ctx: Any,
    ) -> np.ndarray:
        if name not in self._layers:
            raise InvalidFusionError(f"unknown layer: {name}")
        matrix = np.asarray(self._layers[name].matrix(documents, **ctx), dtype=np.float64)
        if item_ids is None:
            item_ids = [document.item_id for document in documents]
        if not item_ids:
            item_ids = [f"item_{index}" for index in range(matrix.shape[0])]
        try:
            validate_similarity_matrix(matrix, item_ids)
        except ValueError as exc:
            raise InvalidFusionError(str(exc)) from exc
        return matrix

    def compute_many(
        self,
        names: Sequence[str],
        documents: Sequence[Document] = (),
        *,
        item_ids: Sequence[str] | None = None,
        **ctx: Any,
    ) -> dict[str, np.ndarray]:
        return {
            name: self.compute(name, documents, item_ids=item_ids, **ctx)
            for name in names
        }
