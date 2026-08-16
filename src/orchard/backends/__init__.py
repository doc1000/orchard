"""Pluggable feature and similarity backends for OrchardBuilder."""

from orchard.backends.fusion import (
    SimilarityProfile,
    fuse_to_dissimilarity,
    raw_convex_fusion,
    validate_dissimilarity,
    variance_calibrated_fusion,
)
from orchard.backends.layers import (
    LayerRegistry,
    MatrixLayer,
    SimilarityLayer,
    TaxonomyCosineLayer,
    TaxonomyJsLayer,
    TfidfCosineLayer,
)
from orchard.backends.minilm import (
    MiniLMCosineLayer,
    MiniLMEmbeddingBackend,
    apply_description_transform,
    mapped_cosine,
    minilm_similarity_matrix,
    raw_cosine,
)
from orchard.backends.similarity import (
    cosine_matrix,
    jensen_shannon_matrix,
    linkage_from_dissimilarity,
    linkage_from_similarity,
    similarity_to_dissimilarity,
    validate_similarity_matrix,
)
from orchard.backends.tfidf import TfidfEmbeddingBackend, tfidf_matrix

__all__ = [
    "LayerRegistry",
    "MatrixLayer",
    "MiniLMCosineLayer",
    "MiniLMEmbeddingBackend",
    "SimilarityLayer",
    "SimilarityProfile",
    "TaxonomyCosineLayer",
    "TaxonomyJsLayer",
    "TfidfCosineLayer",
    "TfidfEmbeddingBackend",
    "apply_description_transform",
    "cosine_matrix",
    "fuse_to_dissimilarity",
    "jensen_shannon_matrix",
    "linkage_from_dissimilarity",
    "linkage_from_similarity",
    "mapped_cosine",
    "minilm_similarity_matrix",
    "raw_convex_fusion",
    "raw_cosine",
    "similarity_to_dissimilarity",
    "tfidf_matrix",
    "validate_dissimilarity",
    "validate_similarity_matrix",
    "variance_calibrated_fusion",
]
