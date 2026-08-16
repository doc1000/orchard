"""ModernBERT taxonomy features (D-024 / Phase 2B).

Source: tool-tree-demo/adapters/transformer_classifier_bridge.py
  - feature_model answerdotai/ModernBERT-base
  - batch 16, seed 20260725
  - LogisticRegression(C=0.1, class_weight=balanced, max_iter=1000)
Source: doc-enrichment transformer_cat/features.py extract_features
  - attention-mask mean pool, max_length=512, 768-d
  - formulas copied; do not import doc-enrichment / transformer_cat

ModernBERT is taxonomy features only. Do not use it as the description
embedding (that remains MiniLM / TF-IDF).
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

from orchard.exceptions import MissingOptionalDependencyError

MODERNBERT_MODEL_ID = "answerdotai/ModernBERT-base"
MODERNBERT_REVISION = "8949b909ec900327062f0ebf497f51aef5e6f0c8"
MODERNBERT_DIMENSIONS = 768
MODERNBERT_SEED = 20260725
TAXONOMY_ML_EXTRA = "taxonomy-ml"

FEATURE_CONFIG = {
    "configuration_version": "modernbert_mean_pool_v1",
    "pooling": "attention-mask-aware mean pooling",
    "batch_size": 16,
    "max_length": 512,
    "dimensions": MODERNBERT_DIMENSIONS,
    "random_seed": MODERNBERT_SEED,
}

EncodeFn = Callable[[Sequence[str]], np.ndarray]


def taxonomy_ml_extra_available() -> bool:
    """True when orchard[taxonomy-ml] imports (torch + transformers) are present."""
    return (
        importlib.util.find_spec("torch") is not None
        and importlib.util.find_spec("transformers") is not None
    )


def missing_taxonomy_ml_error() -> MissingOptionalDependencyError:
    return MissingOptionalDependencyError(
        TAXONOMY_ML_EXTRA,
        "orchard[taxonomy-ml] is required for the ModernBERT taxonomy heads. "
        "Install that extra, or opt in with allow_offline_fallback=True "
        "or an explicit cue-only / custom taxonomy.",
    )


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def default_cache_dir() -> Path:
    override = os.environ.get("ORCHARD_MODERNBERT_CACHE")
    if override:
        return Path(override)
    return Path.home() / ".cache" / "orchard" / "modernbert"


def modernbert_snapshot_available(
    model_id: str = MODERNBERT_MODEL_ID,
    revision: str = MODERNBERT_REVISION,
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


def modernbert_cache_key(
    texts: Sequence[str],
    *,
    model_id: str = MODERNBERT_MODEL_ID,
    revision: str = MODERNBERT_REVISION,
    configuration: dict[str, Any] | None = None,
) -> str:
    """Cache key = model + revision + text checksums + encode config (D-024).

    The Phase 3 ModernBERT bridge omitted revision from its cache spec.
    Orchard includes it.
    """
    text_sha256 = [_sha256_bytes(text.encode("utf-8")) for text in texts]
    cache_spec = {
        "schema_version": "orchard_modernbert_feature_cache_v1",
        "text_sha256": text_sha256,
        "model": model_id,
        "model_revision": revision,
        "configuration": dict(configuration or FEATURE_CONFIG),
    }
    return _sha256_bytes(_canonical_json(cache_spec).encode("utf-8"))


@dataclass
class ModernBERTFeatureBackend:
    """Revision-pinned ModernBERT encoder for taxonomy heads only."""

    model_id: str = MODERNBERT_MODEL_ID
    revision: str = MODERNBERT_REVISION
    pooling: str = FEATURE_CONFIG["pooling"]
    batch_size: int = FEATURE_CONFIG["batch_size"]
    max_length: int = FEATURE_CONFIG["max_length"]
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
            "dimensions": MODERNBERT_DIMENSIONS,
            "random_seed": MODERNBERT_SEED,
            "injected": self.encode_fn is not None,
        }

    def _encode_with_transformers(self, texts: Sequence[str]) -> np.ndarray:
        if not taxonomy_ml_extra_available():
            raise missing_taxonomy_ml_error()
        import torch
        from transformers import AutoModel, AutoTokenizer

        configuration = {
            **FEATURE_CONFIG,
            "batch_size": self.batch_size,
            "max_length": self.max_length,
        }
        cache_dir = Path(self.cache_dir) if self.cache_dir else default_cache_dir()
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_key = modernbert_cache_key(
            texts,
            model_id=self.model_id,
            revision=self.revision,
            configuration=configuration,
        )
        cache_path = (cache_dir / f"{cache_key}.npz").resolve()
        if not cache_path.is_relative_to(cache_dir.resolve()):
            raise ValueError("feature path escapes cache directory")
        text_sha256 = [_sha256_bytes(text.encode("utf-8")) for text in texts]
        if cache_path.exists():
            with np.load(cache_path, allow_pickle=False) as archive:
                features = np.asarray(archive["features"], dtype=np.float32)
                cached_text_sha256 = archive["text_sha256"].astype(str).tolist()
            if cached_text_sha256 != text_sha256:
                raise ValueError("feature cache identity mismatch")
        else:
            seed = int(configuration["random_seed"])
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.use_deterministic_algorithms(True, warn_only=True)
            local_only = (
                modernbert_snapshot_available(self.model_id, self.revision)
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
                # Attention-mask mean pool (transformer_cat.features.extract_features).
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
            raise ValueError(f"unexpected or invalid feature shape: {features.shape}")
        return np.asarray(features, dtype=np.float64)
