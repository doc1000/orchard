"""App-family exact-match similarity layer (D-027, Phase 5).

Source: tool-tree-demo/src/tool_tree_demo/phase3.py ``app_matrix``.
``1.0`` when ``Document.metadata[family_metadata_key]`` is equal and
non-empty, else ``0.0``. Unit diagonal. Never drop this layer and
renormalize: partial metadata is a loud error.

Layer name is ``app_exact_match``. The metadata key is caller-chosen
(``app_name``, ``family``, or another string).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from orchard.document import Document
from orchard.exceptions import InvalidFusionError

FAMILY_LAYER_NAME = "app_exact_match"
DEFAULT_FAMILY_METADATA_KEY = "app_name"


def document_family_value(document: Document, key: str) -> str | None:
    """Return a non-empty family token, or None if missing/blank."""
    if key not in document.metadata:
        return None
    raw = document.metadata[key]
    if raw is None:
        return None
    text = str(raw).strip()
    return text or None


def missing_family_item_ids(
    documents: Sequence[Document],
    key: str,
) -> list[str]:
    """Item ids with a missing or empty family-metadata value."""
    return [
        document.item_id
        for document in documents
        if document_family_value(document, key) is None
    ]


def require_family_metadata(documents: Sequence[Document], key: str) -> None:
    """Abort if any document is missing a non-empty family key.

    Do not drop ``app_exact_match`` and renormalize (D-O-002 / D-O-008).
    """
    missing = missing_family_item_ids(documents, key)
    if missing:
        raise InvalidFusionError(
            f"{FAMILY_LAYER_NAME} requires a non-empty metadata[{key!r}] "
            f"on every document; missing or empty for item ids: {missing}"
        )


@dataclass
class AppExactMatchLayer:
    """Exact family equality. Port of phase3.app_matrix (D-027)."""

    metadata_key: str
    name: str = FAMILY_LAYER_NAME

    def matrix(self, documents: Sequence[Document], **ctx: Any) -> np.ndarray:
        require_family_metadata(documents, self.metadata_key)
        values = np.asarray(
            [
                document_family_value(document, self.metadata_key)
                for document in documents
            ]
        )
        matrix = (values[:, None] == values[None, :]).astype(np.float64)
        np.fill_diagonal(matrix, 1.0)
        return matrix
