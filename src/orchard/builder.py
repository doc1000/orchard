"""OrchardBuilder: construct multi-tree Orchards from documents."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from orchard.assets.profiles import load_semantic_weights
from orchard.backends.fusion import (
    FUSION_MODES,
    SimilarityProfile,
    fuse_to_dissimilarity,
)
from orchard.backends.layers import (
    LayerRegistry,
    TaxonomyCosineLayer,
    TaxonomyJsLayer,
    TfidfCosineLayer,
)
from orchard.backends.minilm import (
    DESCRIPTION_TRANSFORMS,
    MINILM_MODEL_ID,
    MINILM_REVISION,
    SENTENCE_CONFIG,
    MiniLMCosineLayer,
    MiniLMEmbeddingBackend,
    embeddings_extra_available,
    layer_name_for_transform,
    missing_embeddings_error,
)
from orchard.backends.similarity import (
    cosine_matrix,
    jensen_shannon_matrix,
    linkage_from_dissimilarity,
    linkage_from_similarity,
    validate_similarity_matrix,
)
from orchard.backends.tfidf import TfidfEmbeddingBackend
from orchard.document import Document
from orchard.exceptions import InvalidFusionError, InvalidIdentityError
from orchard.identity import ensure_unique_item_ids, validate_tree_id
from orchard.orchard import Orchard
from orchard.taxonomy import Taxonomy, default_taxonomies
from orchard.tree import Tree

_UNSET: Any = object()


def normalize_documents(
    documents: Sequence[Document | str | Mapping[str, Any]],
) -> list[Document]:
    """Normalize builder inputs to canonical Document records."""
    normalized: list[Document] = []
    for index, value in enumerate(documents):
        if isinstance(value, Document):
            normalized.append(value)
        elif isinstance(value, str):
            normalized.append(Document(text=value))
        elif isinstance(value, Mapping):
            normalized.append(Document.from_mapping(value))
        else:
            raise InvalidIdentityError(
                f"unsupported document input at index {index}: {type(value)!r}"
            )
    ensure_unique_item_ids(doc.item_id for doc in normalized)
    if not normalized:
        raise InvalidIdentityError("build requires a non-empty document corpus")
    return normalized


@dataclass
class OrchardBuilder:
    """Build an Orchard from a finite corpus.

    Branching:
    - default / ``None`` taxonomies → packaged Domain + Function trees (cue/JS).
    - ``taxonomies == []`` → one ``semantic`` tree. Default is MiniLM-centered
      cosine fused with document TF-IDF (0.66 / 0.34, variance_calibrated) when
      ``orchard[embeddings]`` is installed or a MiniLM backend is injected.
      TF-IDF-only is explicit: ``TfidfEmbeddingBackend()`` or
      ``allow_offline_fallback=True``.
    - explicit non-empty ``taxonomies`` → one named tree per taxonomy.

    No fused Domain+Function+semantic default tree.
    Public construction verb is ``build`` (not ``fit_transform``).
    """

    taxonomies: Any = field(default=_UNSET)
    embedding_backend: Any | None = None
    lexical_backend: Any | None = None
    linkage_method: str = "average"
    semantic_signed_cosine: bool = False
    taxonomy_similarity: str = "jensen_shannon"
    include_semantic_with_taxonomies: bool = False
    fusion_mode: Literal["variance_calibrated", "raw_convex"] = "variance_calibrated"
    profiles: Mapping[str, SimilarityProfile | Mapping[str, float]] | None = None
    semantic_weights: Mapping[str, float] | None = None
    taxonomy_weights: Mapping[str, Mapping[str, float]] | None = None
    allow_offline_fallback: bool = False
    description_transform: str = "centered"
    metadata: dict[str, Any] = field(default_factory=dict)
    _offline_fallback: bool = field(init=False, repr=False, default=False)

    def __post_init__(self) -> None:
        if self.taxonomy_similarity not in {"jensen_shannon", "cosine"}:
            raise InvalidIdentityError(
                "taxonomy_similarity must be 'jensen_shannon' or 'cosine'"
            )
        if self.fusion_mode not in FUSION_MODES:
            raise InvalidFusionError(
                "fusion_mode must be 'variance_calibrated' or 'raw_convex'"
            )
        if self.description_transform not in DESCRIPTION_TRANSFORMS:
            raise InvalidIdentityError(
                "description_transform must be one of: "
                + ", ".join(DESCRIPTION_TRANSFORMS)
            )
        if self.taxonomies is _UNSET or self.taxonomies is None:
            self.taxonomies = default_taxonomies()
        else:
            self.taxonomies = list(self.taxonomies)
        names = [validate_tree_id(taxonomy.name) for taxonomy in self.taxonomies]
        if len(names) != len(set(names)):
            raise InvalidIdentityError("taxonomy names must be unique tree ids")
        if self.lexical_backend is None:
            self.lexical_backend = TfidfEmbeddingBackend()
        self._resolve_embedding_backend()
        self.profiles = self._resolve_profiles()

    def get_params(self) -> dict[str, Any]:
        """Inspectable builder configuration (sklearn-style)."""
        minilm_active = self._uses_minilm_semantic()
        backend = self.embedding_backend
        return {
            "taxonomies": [taxonomy.name for taxonomy in self.taxonomies],
            "embedding_backend": (
                None if backend is None else type(backend).__name__
            ),
            "lexical_backend": type(self.lexical_backend).__name__,
            "linkage_method": self.linkage_method,
            "semantic_signed_cosine": self.semantic_signed_cosine,
            "taxonomy_similarity": self.taxonomy_similarity,
            "include_semantic_with_taxonomies": self.include_semantic_with_taxonomies,
            "fusion_mode": self.fusion_mode,
            "profiles": {
                tree_id: profile.to_dict() for tree_id, profile in self.profiles.items()
            },
            "active_layers": {
                tree_id: list(profile.weights)
                for tree_id, profile in self.profiles.items()
            },
            "semantic_weights": (
                None if self.semantic_weights is None else dict(self.semantic_weights)
            ),
            "taxonomy_weights": (
                None
                if self.taxonomy_weights is None
                else {
                    name: dict(weights)
                    for name, weights in self.taxonomy_weights.items()
                }
            ),
            "allow_offline_fallback": self.allow_offline_fallback,
            "description_transform": self.description_transform,
            "offline_fallback": self._offline_fallback,
            "model_id": (
                getattr(backend, "model_id", MINILM_MODEL_ID) if minilm_active else None
            ),
            "model_revision": (
                getattr(backend, "revision", MINILM_REVISION) if minilm_active else None
            ),
            "pooling": (
                getattr(backend, "pooling", SENTENCE_CONFIG["pooling"])
                if minilm_active
                else None
            ),
            "metadata": dict(self.metadata),
        }

    def build(
        self,
        documents: Sequence[Document | str | Mapping[str, Any]],
    ) -> Orchard:
        docs = normalize_documents(documents)
        trees: dict[str, Tree] = {}

        if not self.taxonomies:
            trees["semantic"] = self._build_semantic_tree(docs)
        else:
            for taxonomy in self.taxonomies:
                trees[taxonomy.name] = self._build_taxonomy_tree(docs, taxonomy)
            if self.include_semantic_with_taxonomies:
                trees["semantic"] = self._build_semantic_tree(docs)

        return Orchard.from_trees(
            documents=docs,
            trees=trees,
            metadata={
                "builder": self.get_params(),
                **self.metadata,
            },
        )

    def _build_semantic_tree(self, documents: Sequence[Document]) -> Tree:
        profile = self.profiles["semantic"]
        if self._is_native_single_layer(profile, "tfidf_cosine"):
            backend = self._tfidf_encode_backend()
            texts = [doc.text for doc in documents]
            features = np.asarray(backend.encode(texts), dtype=np.float64)
            if features.shape[0] != len(documents):
                raise InvalidIdentityError(
                    "embedding backend row count must match documents"
                )
            similarity = cosine_matrix(features, signed=self.semantic_signed_cosine)
            return self._tree_from_similarity(
                similarity,
                documents,
                tree_id="semantic",
            )
        return self._tree_from_profile(profile, documents)

    def _build_taxonomy_tree(
        self,
        documents: Sequence[Document],
        taxonomy: Taxonomy,
    ) -> Tree:
        profile = self.profiles[taxonomy.name]
        native_layer = self._native_taxonomy_layer(taxonomy.name)
        if self._is_native_single_layer(profile, native_layer):
            distributions = np.asarray(taxonomy.transform(documents), dtype=np.float64)
            if distributions.shape[0] != len(documents):
                raise InvalidIdentityError(
                    "taxonomy transform row count must match documents"
                )
            if self.taxonomy_similarity == "jensen_shannon":
                similarity = jensen_shannon_matrix(distributions)
            else:
                similarity = cosine_matrix(distributions, signed=False)
            return self._tree_from_similarity(
                similarity,
                documents,
                tree_id=taxonomy.name,
            )
        return self._tree_from_profile(profile, documents)

    def _tree_from_similarity(
        self,
        similarity: np.ndarray,
        documents: Sequence[Document],
        *,
        tree_id: str,
    ) -> Tree:
        item_ids = [doc.item_id for doc in documents]
        validate_similarity_matrix(similarity, item_ids)
        z_matrix = linkage_from_similarity(
            similarity,
            method=self.linkage_method,
        )
        return Tree.from_linkage(
            z_matrix,
            item_ids=item_ids,
            tree_id=tree_id,
            method=self.linkage_method,
            documents=documents,
        )

    def _tree_from_profile(
        self,
        profile: SimilarityProfile,
        documents: Sequence[Document],
    ) -> Tree:
        item_ids = [doc.item_id for doc in documents]
        registry = self._layer_registry()
        matrices = registry.compute_many(
            tuple(profile.weights),
            documents,
            item_ids=item_ids,
        )
        dissimilarity = fuse_to_dissimilarity(
            matrices,
            profile.weights,
            fusion_mode=profile.fusion_mode,
        )
        z_matrix = linkage_from_dissimilarity(
            dissimilarity,
            method=self.linkage_method,
        )
        return Tree.from_linkage(
            z_matrix,
            item_ids=item_ids,
            tree_id=profile.name,
            method=self.linkage_method,
            documents=documents,
        )

    def _layer_registry(self) -> LayerRegistry:
        layers: list[Any] = [
            TfidfCosineLayer(
                backend=self.lexical_backend,
                signed=self.semantic_signed_cosine,
            )
        ]
        if self._uses_minilm_semantic():
            layers.append(
                MiniLMCosineLayer(
                    backend=self.embedding_backend,
                    transform=self.description_transform,
                )
            )
        for taxonomy in self.taxonomies:
            layers.append(TaxonomyJsLayer(taxonomy=taxonomy))
            layers.append(TaxonomyCosineLayer(taxonomy=taxonomy))
        return LayerRegistry(layers)

    def _planned_tree_ids(self) -> list[str]:
        if not self.taxonomies:
            return ["semantic"]
        names = [taxonomy.name for taxonomy in self.taxonomies]
        if self.include_semantic_with_taxonomies:
            names.append("semantic")
        return names

    def _native_taxonomy_layer(self, taxonomy_name: str) -> str:
        if self.taxonomy_similarity == "jensen_shannon":
            return f"{taxonomy_name}_raw_js"
        return f"{taxonomy_name}_cosine"

    def _default_weights(self, tree_id: str) -> dict[str, float]:
        if tree_id == "semantic":
            if self.semantic_weights is not None:
                return dict(self.semantic_weights)
            if self._uses_minilm_semantic():
                return load_semantic_weights(
                    minilm_layer=layer_name_for_transform(self.description_transform)
                )
            return {"tfidf_cosine": 1.0}
        if self.taxonomy_weights is not None and tree_id in self.taxonomy_weights:
            return dict(self.taxonomy_weights[tree_id])
        return {self._native_taxonomy_layer(tree_id): 1.0}

    def _needs_semantic(self) -> bool:
        return (not self.taxonomies) or self.include_semantic_with_taxonomies

    def _is_tfidf_backend(self, backend: Any) -> bool:
        return isinstance(backend, TfidfEmbeddingBackend)

    def _uses_minilm_semantic(self) -> bool:
        if not self._needs_semantic() or self._offline_fallback:
            return False
        return not self._is_tfidf_backend(self.embedding_backend)

    def _tfidf_encode_backend(self) -> Any:
        if self._is_tfidf_backend(self.embedding_backend):
            return self.embedding_backend
        return self.lexical_backend

    def _resolve_embedding_backend(self) -> None:
        explicit = self.embedding_backend
        if not self._needs_semantic():
            if explicit is None:
                self.embedding_backend = TfidfEmbeddingBackend()
            self._offline_fallback = False
            return
        if explicit is None:
            if embeddings_extra_available():
                self.embedding_backend = MiniLMEmbeddingBackend()
                self._offline_fallback = False
                return
            if self.allow_offline_fallback:
                self.embedding_backend = TfidfEmbeddingBackend()
                self._offline_fallback = True
                return
            raise missing_embeddings_error()
        if self._is_tfidf_backend(explicit):
            self._offline_fallback = False
            return
        self._offline_fallback = False

    def _coerce_profile(
        self,
        tree_id: str,
        value: SimilarityProfile | Mapping[str, float],
    ) -> SimilarityProfile:
        if isinstance(value, SimilarityProfile):
            return value
        if isinstance(value, Mapping):
            return SimilarityProfile(
                name=tree_id,
                weights=value,
                fusion_mode=self.fusion_mode,
            )
        raise InvalidFusionError(
            f"profile {tree_id!r} must be a SimilarityProfile or weight mapping"
        )

    def _resolve_profiles(self) -> dict[str, SimilarityProfile]:
        provided = dict(self.profiles) if self.profiles is not None else {}
        resolved: dict[str, SimilarityProfile] = {}
        for tree_id, value in provided.items():
            resolved[str(tree_id)] = self._coerce_profile(str(tree_id), value)
        for tree_id in self._planned_tree_ids():
            if tree_id not in resolved:
                resolved[tree_id] = SimilarityProfile(
                    name=tree_id,
                    weights=self._default_weights(tree_id),
                    fusion_mode=self.fusion_mode,
                )
        return resolved

    @staticmethod
    def _is_native_single_layer(
        profile: SimilarityProfile,
        native_layer: str,
    ) -> bool:
        if len(profile.weights) != 1:
            return False
        layer, weight = next(iter(profile.weights.items()))
        return layer == native_layer and math.isclose(
            weight, 1.0, abs_tol=1e-12, rel_tol=0
        )
