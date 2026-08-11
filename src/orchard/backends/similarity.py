"""Similarity / distance / linkage numeric cores (from tool-tree-demo phase3/4a)."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform


def normalize_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms <= 0) or not np.isfinite(norms).all():
        raise ValueError("feature rows must have positive finite norms")
    return values / norms


def finalize_similarity(matrix: np.ndarray) -> np.ndarray:
    result = np.asarray(matrix, dtype=np.float64)
    result = (result + result.T) / 2.0
    result = np.clip(result, 0.0, 1.0)
    np.fill_diagonal(result, 1.0)
    return result


def cosine_matrix(values: np.ndarray, *, signed: bool = False) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    matrix = normalize_rows(values) @ normalize_rows(values).T
    if signed:
        matrix = (matrix + 1.0) / 2.0
    return finalize_similarity(matrix)


def jensen_shannon_matrix(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or np.any(values < 0) or not np.isfinite(values).all():
        raise ValueError("probability vectors must be a finite non-negative matrix")
    totals = values.sum(axis=1)
    if not np.allclose(totals, 1.0, atol=1e-6, rtol=0):
        raise ValueError("probability vectors must sum to one")
    left = values[:, None, :]
    right = values[None, :, :]
    midpoint = (left + right) / 2.0
    left_ratio = np.ones_like(midpoint)
    right_ratio = np.ones_like(midpoint)
    np.divide(left, midpoint, out=left_ratio, where=left > 0)
    np.divide(right, midpoint, out=right_ratio, where=right > 0)
    left_term = np.where(left > 0, left * np.log(left_ratio), 0.0)
    right_term = np.where(right > 0, right * np.log(right_ratio), 0.0)
    divergence = 0.5 * (left_term.sum(axis=2) + right_term.sum(axis=2))
    return finalize_similarity(1.0 - divergence / math.log(2.0))


def validate_similarity_matrix(matrix: np.ndarray, item_ids: Sequence[str]) -> None:
    expected = (len(item_ids), len(item_ids))
    if matrix.shape != expected:
        raise ValueError(f"matrix shape {matrix.shape} does not match {expected}")
    if len(item_ids) != len(set(item_ids)):
        raise ValueError("item ordering contains duplicate IDs")
    if not np.isfinite(matrix).all():
        raise ValueError("matrix contains non-finite values")
    if np.any(matrix < -1e-12) or np.any(matrix > 1 + 1e-12):
        raise ValueError("matrix is outside [0, 1]")
    if not np.allclose(matrix, matrix.T, atol=1e-12, rtol=0):
        raise ValueError("matrix is not symmetric")
    if not np.allclose(np.diag(matrix), 1.0, atol=1e-12, rtol=0):
        raise ValueError("matrix diagonal is not one")


def similarity_to_dissimilarity(matrix: np.ndarray) -> np.ndarray:
    """D = 1 - S with zero diagonal (phase4a raw-convex path)."""
    result = 1.0 - np.asarray(matrix, dtype=np.float64)
    np.fill_diagonal(result, 0.0)
    if (
        result.shape[0] != result.shape[1]
        or not np.isfinite(result).all()
        or not np.allclose(result, result.T, atol=1e-12)
        or np.any(result < -1e-12)
        or not np.allclose(np.diag(result), 0.0, atol=1e-12)
    ):
        raise ValueError("similarity matrix cannot be consumed as a dissimilarity")
    result[result < 0] = 0
    return result


def linkage_from_similarity(
    similarity: np.ndarray,
    *,
    method: str = "average",
) -> np.ndarray:
    """Build SciPy linkage Z from a finalized similarity matrix."""
    dissimilarity = similarity_to_dissimilarity(similarity)
    n = dissimilarity.shape[0]
    if n < 2:
        return np.zeros((0, 4), dtype=np.float64)
    condensed = squareform(dissimilarity, checks=False)
    return linkage(condensed, method=method)
