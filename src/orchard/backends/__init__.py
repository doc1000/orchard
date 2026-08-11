"""Pluggable feature and similarity backends for OrchardBuilder."""

from orchard.backends.similarity import (
    cosine_matrix,
    jensen_shannon_matrix,
    linkage_from_similarity,
    similarity_to_dissimilarity,
    validate_similarity_matrix,
)
from orchard.backends.tfidf import TfidfEmbeddingBackend, tfidf_matrix

__all__ = [
    "TfidfEmbeddingBackend",
    "cosine_matrix",
    "jensen_shannon_matrix",
    "linkage_from_similarity",
    "similarity_to_dissimilarity",
    "tfidf_matrix",
    "validate_similarity_matrix",
]
