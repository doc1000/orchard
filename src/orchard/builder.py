"""OrchardBuilder: construct multi-tree Orchards from documents."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from orchard.assets.profiles import (
    TAXONOMY_PROFILE_NAMES,
    load_semantic_weights,
    load_taxonomy_weights,
)
from orchard.backends.family import (
    DEFAULT_FAMILY_METADATA_KEY,
    FAMILY_LAYER_NAME,
    AppExactMatchLayer,
    require_family_metadata,
)
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
    LAYER_NAMES,
    MINILM_MODEL_ID,
    MINILM_REVISION,
    SENTENCE_CONFIG,
    MiniLMCosineLayer,
    MiniLMEmbeddingBackend,
    embeddings_extra_available,
    layer_name_for_transform,
    missing_embeddings_error,
)
from orchard.backends.modernbert import (
    FEATURE_CONFIG,
    MODERNBERT_MODEL_ID,
    MODERNBERT_REVISION,
    ModernBERTFeatureBackend,
    taxonomy_ml_extra_available,
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
LAYER_MATRIX_PERSIST_POLICIES = (
    "always",
    "never",
    "below_size_limit",
    "compressed",
)


def _matrix_checksum(matrix: np.ndarray) -> str:
    payload = np.ascontiguousarray(matrix, dtype=np.float64)
    return hashlib.sha256(payload.tobytes()).hexdigest()


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
    - default / ``None`` taxonomies → packaged Domain + Function trees from
      fused no-app profiles (cross-taxonomy JS + MiniLM + TF-IDF,
      ``variance_calibrated``). ``transform()`` is ``modernbert_logistic`` when
      ``orchard[taxonomy-ml]`` is installed or ``taxonomy_classifier_backend``
      is injected. Default fused profiles also need ``orchard[embeddings]`` or
      an injected MiniLM. Cue/JS-only is explicit:
      ``allow_offline_fallback=True`` or a cue-only / custom taxonomy.
    - ``taxonomies == []`` → one ``semantic`` tree. Default is MiniLM-centered
      cosine fused with document TF-IDF (0.66 / 0.34, variance_calibrated) when
      ``orchard[embeddings]`` is installed or a MiniLM backend is injected.
      TF-IDF-only is explicit: ``TfidfEmbeddingBackend()`` or
      ``allow_offline_fallback=True``.
    - explicit non-empty ``taxonomies`` → one named tree per taxonomy. Packaged
      Domain/Function heads still fuse when MiniLM is active; other custom
      taxonomies stay single-layer JS unless the caller overrides the profile.

    There is no orchard tree named ``fused`` / ``mixed``. Per-tree fusion is
    how each named tree's matrix is made. Variance-calibrated fusion needs
    n≥3 documents (a selected layer with off-diagonal variance ≤ 1e-12 aborts).
    ``family_metadata_key`` opts into ``app_exact_match`` when every document
    has a non-empty value; partial metadata is a loud error. Public
    construction verb is ``build`` (not ``fit_transform``).
    """

    taxonomies: Any = field(default=_UNSET)
    embedding_backend: Any | None = None
    lexical_backend: Any | None = None
    taxonomy_classifier_backend: Any | None = None
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
    family_metadata_key: str | None = None
    layer_matrix_persist: Literal[
        "always", "never", "below_size_limit", "compressed"
    ] = "never"
    metadata: dict[str, Any] = field(default_factory=dict)
    _offline_fallback: bool = field(init=False, repr=False, default=False)
    _default_taxonomy_build: bool = field(init=False, repr=False, default=False)
    _app_exact_match_active: bool = field(init=False, repr=False, default=False)
    _base_profiles: dict[str, SimilarityProfile] = field(
        init=False, repr=False, default_factory=dict
    )
    _caller_profile_tree_ids: set[str] = field(
        init=False, repr=False, default_factory=set
    )
    _family_lookup_key: str | None = field(init=False, repr=False, default=None)

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
        if self.family_metadata_key is not None and not str(
            self.family_metadata_key
        ).strip():
            raise InvalidIdentityError(
                "family_metadata_key must be a non-empty string or None"
            )
        if self.layer_matrix_persist not in LAYER_MATRIX_PERSIST_POLICIES:
            raise InvalidIdentityError(
                "layer_matrix_persist must be one of: "
                + ", ".join(LAYER_MATRIX_PERSIST_POLICIES)
            )
        self._default_taxonomy_build = self.taxonomies is _UNSET or self.taxonomies is None
        if self.taxonomies is _UNSET or self.taxonomies is None:
            if self.taxonomy_classifier_backend is None and taxonomy_ml_extra_available():
                self.taxonomy_classifier_backend = ModernBERTFeatureBackend()
            self.taxonomies = default_taxonomies(
                allow_offline_fallback=self.allow_offline_fallback,
                classifier_backend=self.taxonomy_classifier_backend,
            )
        else:
            self.taxonomies = list(self.taxonomies)
        names = [validate_tree_id(taxonomy.name) for taxonomy in self.taxonomies]
        if len(names) != len(set(names)):
            raise InvalidIdentityError("taxonomy names must be unique tree ids")
        if self.lexical_backend is None:
            self.lexical_backend = TfidfEmbeddingBackend()
        self._resolve_taxonomy_classifier_backend()
        self._resolve_embedding_backend()
        self._caller_profile_tree_ids = self._caller_override_tree_ids()
        self.profiles = self._resolve_profiles()
        self._base_profiles = dict(self.profiles)

    def get_params(self) -> dict[str, Any]:
        """Inspectable builder configuration (sklearn-style)."""
        minilm_active = self._uses_minilm()
        backend = self.embedding_backend
        taxonomy_backend = self.taxonomy_classifier_backend
        taxonomy_head_active = self._uses_modernbert_taxonomy()
        return {
            "taxonomies": [taxonomy.name for taxonomy in self.taxonomies],
            "embedding_backend": (
                None if backend is None else type(backend).__name__
            ),
            "taxonomy_classifier_backend": (
                None if taxonomy_backend is None else type(taxonomy_backend).__name__
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
            "family_metadata_key": self.family_metadata_key,
            "app_exact_match_active": self._app_exact_match_active,
            "layer_matrix_persist": self.layer_matrix_persist,
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
            "taxonomy_transform": self._taxonomy_transform_kind(),
            "taxonomy_model_id": (
                getattr(taxonomy_backend, "model_id", MODERNBERT_MODEL_ID)
                if taxonomy_head_active
                else None
            ),
            "taxonomy_model_revision": (
                getattr(taxonomy_backend, "revision", MODERNBERT_REVISION)
                if taxonomy_head_active
                else None
            ),
            "taxonomy_pooling": (
                getattr(taxonomy_backend, "pooling", FEATURE_CONFIG["pooling"])
                if taxonomy_head_active
                else None
            ),
            "metadata": dict(self.metadata),
        }

    def build(
        self,
        documents: Sequence[Document | str | Mapping[str, Any]],
    ) -> Orchard:
        docs = normalize_documents(documents)
        self.profiles = dict(self._base_profiles)
        self._prepare_family_layer(docs)
        layer_matrices = self._compute_shared_layers(docs)
        trees: dict[str, Tree] = {}

        if not self.taxonomies:
            trees["semantic"] = self._build_semantic_tree(docs, layer_matrices)
        else:
            for taxonomy in self.taxonomies:
                trees[taxonomy.name] = self._build_taxonomy_tree(
                    docs, taxonomy, layer_matrices
                )
            if self.include_semantic_with_taxonomies:
                trees["semantic"] = self._build_semantic_tree(docs, layer_matrices)

        return Orchard.from_trees(
            documents=docs,
            trees=trees,
            metadata={
                **self.metadata,
                "builder": self.get_params(),
                "layer_checksums": {
                    name: _matrix_checksum(matrix)
                    for name, matrix in layer_matrices.items()
                },
                "fusion_mode": self.fusion_mode,
                "profiles": {
                    tree_id: self.profiles[tree_id].to_dict() for tree_id in trees
                },
                "taxonomy_transform": self._taxonomy_transform_kind(),
                "family_metadata_key": self.family_metadata_key,
                "app_exact_match_active": self._app_exact_match_active,
                "layer_matrix_persist": self.layer_matrix_persist,
            },
            layer_matrices=layer_matrices,
        )

    def _build_semantic_tree(
        self,
        documents: Sequence[Document],
        layer_matrices: dict[str, np.ndarray],
    ) -> Tree:
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
            layer_matrices.setdefault("tfidf_cosine", similarity)
            return self._tree_from_similarity(
                similarity,
                documents,
                tree_id="semantic",
            )
        return self._tree_from_profile(profile, documents, layer_matrices)

    def _build_taxonomy_tree(
        self,
        documents: Sequence[Document],
        taxonomy: Taxonomy,
        layer_matrices: dict[str, np.ndarray],
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
            layer_matrices.setdefault(native_layer, similarity)
            return self._tree_from_similarity(
                similarity,
                documents,
                tree_id=taxonomy.name,
            )
        return self._tree_from_profile(profile, documents, layer_matrices)

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
        layer_matrices: Mapping[str, np.ndarray],
    ) -> Tree:
        missing = [name for name in profile.weights if name not in layer_matrices]
        if missing:
            raise InvalidFusionError(
                f"profile {profile.name!r} missing shared layers: {missing}"
            )
        matrices = {name: layer_matrices[name] for name in profile.weights}
        dissimilarity = fuse_to_dissimilarity(
            matrices,
            profile.weights,
            fusion_mode=profile.fusion_mode,
        )
        item_ids = [doc.item_id for doc in documents]
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

    def _compute_shared_layers(
        self,
        documents: Sequence[Document],
    ) -> dict[str, np.ndarray]:
        needed: list[str] = []
        for tree_id in self._planned_tree_ids():
            profile = self.profiles[tree_id]
            native = (
                "tfidf_cosine"
                if tree_id == "semantic"
                else self._native_taxonomy_layer(tree_id)
            )
            if not self._is_native_single_layer(profile, native):
                needed.extend(profile.weights)
        if not needed:
            return {}
        registry = self._layer_registry()
        item_ids = [document.item_id for document in documents]
        matrices: dict[str, np.ndarray] = {}
        for name in dict.fromkeys(needed):
            matrices[name] = registry.compute(name, documents, item_ids=item_ids)
        return matrices

    def _layer_registry(self) -> LayerRegistry:
        layers: list[Any] = [
            TfidfCosineLayer(
                backend=self.lexical_backend,
                signed=self.semantic_signed_cosine,
            )
        ]
        if self._uses_minilm():
            layers.append(
                MiniLMCosineLayer(
                    backend=self.embedding_backend,
                    transform=self.description_transform,
                )
            )
        if self._app_exact_match_active and self._family_lookup_key:
            layers.append(AppExactMatchLayer(metadata_key=self._family_lookup_key))
        for taxonomy in self.taxonomies:
            layers.append(TaxonomyJsLayer(taxonomy=taxonomy))
            layers.append(TaxonomyCosineLayer(taxonomy=taxonomy))
        return LayerRegistry(layers)

    def _caller_override_tree_ids(self) -> set[str]:
        names: set[str] = set()
        if self.profiles is not None:
            names.update(str(tree_id) for tree_id in self.profiles)
        if self.taxonomy_weights is not None:
            names.update(str(tree_id) for tree_id in self.taxonomy_weights)
        if self.semantic_weights is not None:
            names.add("semantic")
        return names

    def _effective_family_key(self) -> str:
        if self.family_metadata_key:
            return str(self.family_metadata_key)
        return DEFAULT_FAMILY_METADATA_KEY

    def _can_auto_enable_family(self) -> bool:
        return self._uses_minilm() and not self._offline_fallback

    def _will_use_family_layer(self) -> bool:
        for tree_id in self._planned_tree_ids():
            profile = self.profiles.get(tree_id)
            if profile is not None and FAMILY_LAYER_NAME in profile.weights:
                return True
        if not (self.family_metadata_key and self._can_auto_enable_family()):
            return False
        return any(
            tree_id in TAXONOMY_PROFILE_NAMES
            and tree_id not in self._caller_profile_tree_ids
            for tree_id in self._planned_tree_ids()
        )

    def _apply_with_app_defaults(self) -> None:
        if not (self.family_metadata_key and self._can_auto_enable_family()):
            return
        taxonomy_names = [taxonomy.name for taxonomy in self.taxonomies]
        minilm_layer = layer_name_for_transform(self.description_transform)
        for tree_id in self._planned_tree_ids():
            if tree_id == "semantic":
                continue
            if tree_id in self._caller_profile_tree_ids:
                continue
            if tree_id not in TAXONOMY_PROFILE_NAMES or not self._uses_minilm():
                continue
            self.profiles[tree_id] = SimilarityProfile(
                name=tree_id,
                weights=load_taxonomy_weights(
                    tree_id,
                    taxonomy_names=taxonomy_names,
                    minilm_layer=minilm_layer,
                    with_app=True,
                ),
                fusion_mode=self.fusion_mode,
            )

    def _prepare_family_layer(self, documents: Sequence[Document]) -> None:
        self._app_exact_match_active = False
        self._family_lookup_key = None
        if not self._will_use_family_layer():
            return
        key = self._effective_family_key()
        require_family_metadata(documents, key)
        self._apply_with_app_defaults()
        self._family_lookup_key = key
        self._app_exact_match_active = any(
            FAMILY_LAYER_NAME in self.profiles[tree_id].weights
            for tree_id in self._planned_tree_ids()
            if tree_id in self.profiles
        )

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
            if self._uses_minilm():
                return load_semantic_weights(
                    minilm_layer=layer_name_for_transform(self.description_transform)
                )
            return {"tfidf_cosine": 1.0}
        if self.taxonomy_weights is not None and tree_id in self.taxonomy_weights:
            return dict(self.taxonomy_weights[tree_id])
        if self._uses_minilm() and tree_id in TAXONOMY_PROFILE_NAMES:
            return load_taxonomy_weights(
                tree_id,
                taxonomy_names=[taxonomy.name for taxonomy in self.taxonomies],
                minilm_layer=layer_name_for_transform(self.description_transform),
            )
        return {self._native_taxonomy_layer(tree_id): 1.0}

    def _needs_semantic(self) -> bool:
        return (not self.taxonomies) or self.include_semantic_with_taxonomies

    def _is_tfidf_backend(self, backend: Any) -> bool:
        return isinstance(backend, TfidfEmbeddingBackend)

    def _uses_minilm(self) -> bool:
        if self._offline_fallback:
            return False
        return not self._is_tfidf_backend(self.embedding_backend)

    def _uses_minilm_semantic(self) -> bool:
        return self._uses_minilm()

    def _requires_minilm(self) -> bool:
        if self._needs_semantic():
            return True
        if self._default_taxonomy_build:
            return True
        if self._uses_modernbert_taxonomy():
            return True
        return False

    def _profile_uses_minilm_layer(self, profile: SimilarityProfile) -> bool:
        return bool(set(profile.weights) & set(LAYER_NAMES.values()))

    def _tfidf_encode_backend(self) -> Any:
        if self._is_tfidf_backend(self.embedding_backend):
            return self.embedding_backend
        return self.lexical_backend

    def _taxonomy_transform_kind(self) -> str:
        kinds = [
            str(getattr(taxonomy, "taxonomy_transform", "cue"))
            for taxonomy in self.taxonomies
        ]
        if "modernbert_logistic" in kinds:
            return "modernbert_logistic"
        return "cue"

    def _uses_modernbert_taxonomy(self) -> bool:
        return self._taxonomy_transform_kind() == "modernbert_logistic"

    def _resolve_taxonomy_classifier_backend(self) -> None:
        if self.taxonomy_classifier_backend is not None:
            return
        for taxonomy in self.taxonomies:
            encoder = getattr(taxonomy, "feature_encoder", None)
            if encoder is not None:
                self.taxonomy_classifier_backend = encoder
                return
        if self._uses_modernbert_taxonomy():
            self.taxonomy_classifier_backend = ModernBERTFeatureBackend()

    def _resolve_embedding_backend(self) -> None:
        explicit = self.embedding_backend
        if explicit is not None:
            self._offline_fallback = False
            return
        if self._requires_minilm():
            if embeddings_extra_available():
                self.embedding_backend = MiniLMEmbeddingBackend()
                self._offline_fallback = False
                return
            if self.allow_offline_fallback:
                self.embedding_backend = TfidfEmbeddingBackend()
                self._offline_fallback = True
                return
            raise missing_embeddings_error()
        self.embedding_backend = TfidfEmbeddingBackend()
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
        self._assert_minilm_layers_available(resolved)
        return resolved

    def _assert_minilm_layers_available(
        self,
        profiles: Mapping[str, SimilarityProfile],
    ) -> None:
        if self._uses_minilm():
            return
        if any(self._profile_uses_minilm_layer(profile) for profile in profiles.values()):
            raise missing_embeddings_error()

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
