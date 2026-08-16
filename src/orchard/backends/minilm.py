"""MiniLM description embeddings and Phase 3B transforms (D-029 / D-030).

Source: tool-tree-demo/src/tool_tree_demo/phase3b.py
  - build_description_variants
  - _raw_cosine
  - _mapped_cosine
  - TRANSFORM_CONFIG
Source: tool-tree-demo/adapters/sentence_embedding_bridge.py
  - in-process encode: revision, seed 20260725, batch 32, max_length 256,
    attention-mask mean pool, cache key = model+revision+text sha256+config

Loaded with transformers.AutoModel / AutoTokenizer only. Do not add the
sentence-transformers package. Centering / whitening / PC-3 happen after
encode in numpy, never inside the encoder.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from orchard.document import Document
from orchard.exceptions import InvalidFusionError, MissingOptionalDependencyError

MINILM_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MINILM_REVISION = "c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
MINILM_DIMENSIONS = 384
MINILM_SEED = 20260725
EMBEDDINGS_EXTRA = "embeddings"

SENTENCE_CONFIG = {
    "configuration_version": "minilm_mean_pool_v1",
    "pooling": "attention-mask-aware mean pooling",
    "transformation_version": "phase3b_embedding_transforms_v1",
    "batch_size": 32,
    "max_length": 256,
    "dimensions": MINILM_DIMENSIONS,
    "random_seed": MINILM_SEED,
}
TRANSFORM_CONFIG = {
    "transformation_version": "phase3b_embedding_transforms_v1",
    "cosine_mapping": "(raw_cosine + 1) / 2 after raw off-diagonal measurement",
    "pca_components_removed": 3,
    "whitening_relative_eigenvalue_floor": 1e-6,
    "whitening_ridge_fraction": 1e-4,
}
DESCRIPTION_TRANSFORMS = ("centered", "raw", "whitened", "centered_pc3")
LAYER_NAMES = {
    "raw": "description_minilm_raw_cosine",
    "centered": "description_minilm_centered_cosine",
    "whitened": "description_minilm_whitened_cosine",
    "centered_pc3": "description_minilm_centered_pc3_cosine",
}

EncodeFn = Callable[[Sequence[str]], np.ndarray]


def embeddings_extra_available() -> bool:
    """True when orchard[embeddings] imports (torch + transformers) are present."""
    return (
        importlib.util.find_spec("torch") is not None
        and importlib.util.find_spec("transformers") is not None
    )


def missing_embeddings_error() -> MissingOptionalDependencyError:
    return MissingOptionalDependencyError(
        EMBEDDINGS_EXTRA,
        "orchard[embeddings] is required for the MiniLM description layer "
        "(semantic tree and fused taxonomy profiles). "
        "Install that extra, or opt in with allow_offline_fallback=True or "
        "embedding_backend=TfidfEmbeddingBackend().",
    )


def layer_name_for_transform(transform: str) -> str:
    if transform not in LAYER_NAMES:
        raise InvalidFusionError(
            "description_transform must be one of: "
            + ", ".join(DESCRIPTION_TRANSFORMS)
        )
    return LAYER_NAMES[transform]


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _row_normalize(values: np.ndarray) -> np.ndarray:
    # Port of phase3b._row_normalize (D-029).
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms <= 1e-15) or not np.isfinite(norms).all():
        raise ValueError("embedding transform produced invalid row norms")
    return values / norms


def raw_cosine(values: np.ndarray) -> np.ndarray:
    """Port of phase3b._raw_cosine: row-normalize, then symmetrized cosine."""
    normalized = _row_normalize(np.asarray(values, dtype=np.float64))
    result = normalized @ normalized.T
    return (result + result.T) / 2.0


def mapped_cosine(raw: np.ndarray) -> np.ndarray:
    """Port of phase3b._mapped_cosine: clip((S+1)/2, 0, 1), diag=1."""
    result = np.clip((raw + 1.0) / 2.0, 0.0, 1.0)
    np.fill_diagonal(result, 1.0)
    return result


def apply_description_transform(
    raw_features: np.ndarray,
    transform: str = "centered",
) -> np.ndarray:
    """Corpus-level MiniLM transforms (phase3b.build_description_variants).

    raw / centered skip the eigen path. whitened / centered_pc3 use sample
    covariance ``centered.T @ centered / (n-1)``, ``eigh``, descending
    eigenvalues, floor ``1e-6 * λ_max``, ridge ``1e-4 * λ_max``, and drop
    3 PCs only for ``centered_pc3``.
    """
    if transform not in DESCRIPTION_TRANSFORMS:
        raise InvalidFusionError(
            "description_transform must be one of: "
            + ", ".join(DESCRIPTION_TRANSFORMS)
        )
    raw_features = np.asarray(raw_features, dtype=np.float64)
    if raw_features.ndim != 2 or raw_features.shape[0] < 1:
        raise ValueError("MiniLM features must be a non-empty [n, d] matrix")
    if transform == "raw":
        return raw_features
    mean = raw_features.mean(axis=0)
    centered = raw_features - mean
    if transform == "centered":
        return centered
    n_docs = raw_features.shape[0]
    if n_docs < 2:
        raise ValueError("whitening / PC-3 require at least two documents")
    covariance = centered.T @ centered / (n_docs - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    maximum = float(eigenvalues[0])
    floor = maximum * float(TRANSFORM_CONFIG["whitening_relative_eigenvalue_floor"])
    retained = eigenvalues > floor
    retained_count = int(retained.sum())
    if retained_count < 2:
        raise ValueError("whitening retained too few non-negligible components")
    ridge = maximum * float(TRANSFORM_CONFIG["whitening_ridge_fraction"])
    if transform == "whitened":
        return (centered @ eigenvectors[:, retained]) / np.sqrt(
            eigenvalues[retained] + ridge
        )
    removed = int(TRANSFORM_CONFIG["pca_components_removed"])
    return centered - (centered @ eigenvectors[:, :removed]) @ eigenvectors[:, :removed].T


def minilm_similarity_matrix(
    raw_features: np.ndarray,
    *,
    transform: str = "centered",
) -> np.ndarray:
    """Measure raw cosine on the transformed features, then signed-map (S+1)/2."""
    features = apply_description_transform(raw_features, transform)
    return mapped_cosine(raw_cosine(features))


def default_cache_dir() -> Path:
    override = os.environ.get("ORCHARD_EMBEDDING_CACHE")
    if override:
        return Path(override)
    return Path.home() / ".cache" / "orchard" / "minilm"


def minilm_snapshot_available(
    model_id: str = MINILM_MODEL_ID,
    revision: str = MINILM_REVISION,
) -> bool:
    """True when a revision-pinned Hugging Face snapshot is already local."""
    try:
        from huggingface_hub.file_download import (  # type: ignore[import-untyped]
            _CACHED_NO_EXIST,
            try_to_load_from_cache,
        )
    except ImportError:
        return False
    path = try_to_load_from_cache(
        repo_id=model_id,
        filename="config.json",
        revision=revision,
    )
    return bool(path) and path is not _CACHED_NO_EXIST


def embedding_cache_key(
    texts: Sequence[str],
    *,
    model_id: str = MINILM_MODEL_ID,
    revision: str = MINILM_REVISION,
    configuration: dict[str, Any] | None = None,
) -> str:
    """Cache key = model + revision + text checksums + encode config (D-029)."""
    text_sha256 = [_sha256_bytes(text.encode("utf-8")) for text in texts]
    cache_spec = {
        "schema_version": "phase3b_sentence_embedding_cache_v1",
        "text_sha256": text_sha256,
        "model": model_id,
        "model_revision": revision,
        "configuration": dict(configuration or SENTENCE_CONFIG),
    }
    return _sha256_bytes(_canonical_json(cache_spec).encode("utf-8"))


@dataclass
class MiniLMEmbeddingBackend:
    """Revision-pinned MiniLM encoder. Transforms are applied after ``encode``."""

    model_id: str = MINILM_MODEL_ID
    revision: str = MINILM_REVISION
    pooling: str = SENTENCE_CONFIG["pooling"]
    batch_size: int = SENTENCE_CONFIG["batch_size"]
    max_length: int = SENTENCE_CONFIG["max_length"]
    encode_fn: EncodeFn | None = None
    cache_dir: Path | None = None
    local_files_only: bool | None = None

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if self.encode_fn is not None:
            return np.asarray(self.encode_fn(texts), dtype=np.float64)
        return self._encode_with_transformers(texts)

    def provenance(self) -> dict[str, Any]:
        return {
            "backend": type(self).__name__,
            "model_id": self.model_id,
            "revision": self.revision,
            "pooling": self.pooling,
            "batch_size": self.batch_size,
            "max_length": self.max_length,
            "dimensions": MINILM_DIMENSIONS,
            "random_seed": MINILM_SEED,
            "injected": self.encode_fn is not None,
        }

    def _encode_with_transformers(self, texts: Sequence[str]) -> np.ndarray:
        if not embeddings_extra_available():
            raise missing_embeddings_error()
        import torch
        from transformers import AutoModel, AutoTokenizer

        configuration = {
            **SENTENCE_CONFIG,
            "batch_size": self.batch_size,
            "max_length": self.max_length,
        }
        cache_dir = Path(self.cache_dir) if self.cache_dir else default_cache_dir()
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_key = embedding_cache_key(
            texts,
            model_id=self.model_id,
            revision=self.revision,
            configuration=configuration,
        )
        cache_path = (cache_dir / f"{cache_key}.npz").resolve()
        if not cache_path.is_relative_to(cache_dir.resolve()):
            raise ValueError("embedding path escapes cache directory")
        text_sha256 = [_sha256_bytes(text.encode("utf-8")) for text in texts]
        if cache_path.exists():
            with np.load(cache_path, allow_pickle=False) as archive:
                features = np.asarray(archive["features"], dtype=np.float32)
                cached_text_sha256 = archive["text_sha256"].astype(str).tolist()
            if cached_text_sha256 != text_sha256:
                raise ValueError("embedding cache identity mismatch")
        else:
            seed = int(configuration["random_seed"])
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.use_deterministic_algorithms(True, warn_only=True)
            local_only = (
                minilm_snapshot_available(self.model_id, self.revision)
                if self.local_files_only is None
                else self.local_files_only
            )
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                revision=self.revision,
                local_files_only=local_only,
            )
            encoder = AutoModel.from_pretrained(
                self.model_id,
                revision=self.revision,
                local_files_only=local_only,
            )
            encoder.eval()
            rows: list[np.ndarray] = []
            batch_size = int(configuration["batch_size"])
            max_length = int(configuration["max_length"])
            for start in range(0, len(texts), batch_size):
                encoded = tokenizer(
                    texts[start : start + batch_size],
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
                with torch.no_grad():
                    hidden = encoder(**encoded).last_hidden_state
                mask = encoded["attention_mask"].unsqueeze(-1).expand(hidden.size()).float()
                pooled = torch.sum(hidden * mask, dim=1) / torch.clamp(
                    mask.sum(dim=1), min=1e-9
                )
                rows.append(pooled.cpu().numpy().astype(np.float32))
            features = np.concatenate(rows, axis=0)
            np.savez_compressed(
                cache_path,
                features=features,
                text_sha256=np.asarray(text_sha256),
            )
        expected_shape = (len(texts), int(configuration["dimensions"]))
        if features.shape != expected_shape or not np.isfinite(features).all():
            raise ValueError(f"unexpected or invalid embedding shape: {features.shape}")
        return np.asarray(features, dtype=np.float64)


@dataclass
class MiniLMCosineLayer:
    """Independent MiniLM similarity layer (never concatenated with TF-IDF)."""

    backend: Any
    transform: str = "centered"
    name: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            self.name = layer_name_for_transform(self.transform)

    def matrix(self, documents: Sequence[Document], **ctx: Any) -> np.ndarray:
        texts = [document.text for document in documents]
        features = np.asarray(self.backend.encode(texts), dtype=np.float64)
        if features.shape[0] != len(documents):
            raise InvalidFusionError(
                f"{self.name} row count must match documents"
            )
        return minilm_similarity_matrix(features, transform=self.transform)
