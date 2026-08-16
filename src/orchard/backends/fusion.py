"""Layered similarity fusion (AppWorld Phase 3 / 3B numerics).

Source: tool-tree-demo/src/tool_tree_demo/phase3b.py
  - raw_convex_fusion
  - variance_calibrated_fusion
  - validate_dissimilarity
Weight abort rules: tool-tree-demo/src/tool_tree_demo/phase3.py
  - load_profiles / fuse
Decisions: D-027, D-030, D-031.

Do not silently renormalize weights. Do not concatenate feature vectors.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence

import numpy as np

from orchard.exceptions import InvalidFusionError

WEIGHT_ABS_TOL = 1e-12
VARIANCE_FLOOR = 1e-12
FUSION_MODES = ("variance_calibrated", "raw_convex")


def validate_fusion_weights(
    weights: Mapping[str, float],
    available_layers: set[str] | None = None,
) -> dict[str, float]:
    """Abort on empty, non-finite, negative, unknown, or non-unit-sum weights.

    Port of phase3.load_profiles / phase3.fuse weight rules (D-027).
    Never silently renormalizes.
    """
    if not isinstance(weights, Mapping) or not weights:
        raise InvalidFusionError("fusion weights must be a non-empty mapping")
    if available_layers is not None:
        unknown = set(weights) - available_layers
        if unknown:
            raise InvalidFusionError(
                f"unknown fusion layers: {sorted(unknown)}"
            )
    values = {layer: float(weight) for layer, weight in weights.items()}
    if any(not math.isfinite(value) or value < 0 for value in values.values()):
        raise InvalidFusionError("fusion weights must be finite and non-negative")
    if not math.isclose(sum(values.values()), 1.0, abs_tol=WEIGHT_ABS_TOL, rel_tol=0):
        raise InvalidFusionError("fusion weights must sum exactly to one")
    return values


@dataclass(frozen=True)
class SimilarityProfile:
    """Named convex combination of independent similarity layers."""

    name: str
    weights: Mapping[str, float]
    fusion_mode: Literal["variance_calibrated", "raw_convex"] = "variance_calibrated"

    def __post_init__(self) -> None:
        object.__setattr__(self, "weights", validate_fusion_weights(self.weights))
        if self.fusion_mode not in FUSION_MODES:
            raise InvalidFusionError(
                "fusion_mode must be 'variance_calibrated' or 'raw_convex'"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "weights": dict(self.weights),
            "fusion_mode": self.fusion_mode,
        }


def raw_convex_fusion(
    matrices: Mapping[str, np.ndarray], weights: Mapping[str, float]
) -> np.ndarray:
    """Strict convex sum of similarity matrices, then finalize.

    Port of phase3b.raw_convex_fusion (D-027, D-030):
    S = Σ wᵢ Sᵢ; symmetrize; diag=1; clip [0, 1].
    """
    result = np.zeros_like(next(iter(matrices.values())), dtype=np.float64)
    for name, weight in weights.items():
        result += weight * matrices[name]
    result = (result + result.T) / 2.0
    np.fill_diagonal(result, 1.0)
    return np.clip(result, 0.0, 1.0)


def variance_calibrated_fusion(
    matrices: Mapping[str, np.ndarray],
    weights: Mapping[str, float],
    *,
    variance_floor: float = VARIANCE_FLOOR,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Z-score each layer's upper-triangle off-diagonals, then weighted D.

    Port of phase3b.variance_calibrated_fusion (D-030):
    abort if any layer variance or total fused variance <= 1e-12;
    D = max(offdiag fused z) − fused z; symmetrize; max(D, 0); diag=0.
    """
    upper = np.triu_indices_from(next(iter(matrices.values())), k=1)
    standardized: dict[str, np.ndarray] = {}
    layer_stats: dict[str, Any] = {}
    contributions: dict[str, np.ndarray] = {}
    for name, weight in weights.items():
        values = matrices[name][upper]
        mean = float(values.mean())
        std = float(values.std())
        variance = std * std
        if variance <= variance_floor:
            raise InvalidFusionError(
                f"{name} off-diagonal variance is numerically negligible"
            )
        z = (matrices[name] - mean) / std
        standardized[name] = z
        contributions[name] = weight * z[upper]
        layer_stats[name] = {
            "off_diagonal_mean": mean,
            "off_diagonal_std": std,
            "standalone_variance": variance,
            "nominal_weight": weight,
            "effective_similarity_coefficient": weight / std,
            "weighted_standardized_variance": float(contributions[name].var()),
        }
    fused_standardized = sum(
        weight * standardized[name] for name, weight in weights.items()
    )
    fused_upper = fused_standardized[upper]
    total_variance = float(fused_upper.var())
    if total_variance <= variance_floor:
        raise InvalidFusionError("calibrated fused variance is numerically negligible")
    maximum = float(fused_upper.max())
    dissimilarity = maximum - fused_standardized
    dissimilarity = np.maximum((dissimilarity + dissimilarity.T) / 2.0, 0.0)
    np.fill_diagonal(dissimilarity, 0.0)
    names = list(weights)
    covariance = np.cov(
        np.vstack([contributions[name] for name in names]), bias=True
    )
    for name in names:
        marginal = float(np.cov(contributions[name], fused_upper, bias=True)[0, 1])
        layer_stats[name]["approximate_marginal_covariance_with_fused"] = marginal
        layer_stats[name]["approximate_marginal_fraction_of_total_variance"] = (
            marginal / total_variance
        )
    return dissimilarity, {
        "schema_version": "variance_calibrated_fusion_v1",
        "method": (
            "standardize each selected similarity using upper-triangle mean/std, "
            "apply nominal weights, then D=max(offdiag fused z)-fused z"
        ),
        "variance_floor": variance_floor,
        "layers": layer_stats,
        "weighted_contribution_covariance": {
            "layer_order": names,
            "matrix": covariance.tolist(),
        },
        "total_fused_standardized_variance": total_variance,
        "global_dissimilarity_shift": maximum,
        "interpretation_warning": (
            "Marginal covariance fractions overlap for correlated layers and are "
            "not independent variance shares."
        ),
    }


def validate_dissimilarity(matrix: np.ndarray, item_ids: Sequence[str]) -> None:
    """Port of phase3b.validate_dissimilarity (D-030, D-031)."""
    expected = (len(item_ids), len(item_ids))
    if matrix.shape != expected or not np.isfinite(matrix).all():
        raise InvalidFusionError("invalid dissimilarity shape or values")
    if np.any(matrix < -1e-12):
        raise InvalidFusionError("dissimilarity contains negative values")
    if not np.allclose(matrix, matrix.T, atol=1e-12, rtol=0):
        raise InvalidFusionError("dissimilarity is not symmetric")
    if not np.allclose(np.diag(matrix), 0.0, atol=1e-12, rtol=0):
        raise InvalidFusionError("dissimilarity diagonal is not zero")


def fuse_to_dissimilarity(
    matrices: Mapping[str, np.ndarray],
    weights: Mapping[str, float],
    *,
    fusion_mode: Literal["variance_calibrated", "raw_convex"] = "variance_calibrated",
    variance_floor: float = VARIANCE_FLOOR,
) -> np.ndarray:
    """Combine layer matrices and return linkage-ready dissimilarity D.

    raw_convex: S from phase3b.raw_convex_fusion, then D = 1 − S (phase4a).
    variance_calibrated: consume phase3b D as-is (do not feed 1 − D/max(D)).
    """
    validated = validate_fusion_weights(weights, set(matrices))
    if fusion_mode == "raw_convex":
        similarity = raw_convex_fusion(matrices, validated)
        dissimilarity = 1.0 - similarity
        np.fill_diagonal(dissimilarity, 0.0)
        dissimilarity[dissimilarity < 0] = 0
        return dissimilarity
    if fusion_mode == "variance_calibrated":
        dissimilarity, _calibration = variance_calibrated_fusion(
            matrices,
            validated,
            variance_floor=variance_floor,
        )
        return dissimilarity
    raise InvalidFusionError(
        "fusion_mode must be 'variance_calibrated' or 'raw_convex'"
    )
