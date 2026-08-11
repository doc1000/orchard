"""Taxonomy protocols and fixture stubs (full Domain/Function artifacts in Phase 3)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Protocol, Sequence, runtime_checkable

import numpy as np

from orchard.document import Document
from orchard.exceptions import InvalidIdentityError
from orchard.identity import validate_tree_id


@runtime_checkable
class Taxonomy(Protocol):
    """Semantic classification structure used as a per-tree build signal."""

    @property
    def name(self) -> str:
        """Tree id / taxonomy name (e.g. ``domain``, ``function``)."""

    @property
    def label_order(self) -> tuple[str, ...]:
        """Stable label axis for distribution columns."""

    def transform(self, documents: Sequence[Document]) -> np.ndarray:
        """Return ``(n_docs, n_labels)`` probability rows summing to 1."""


@dataclass
class StubTaxonomy:
    """Deterministic fixture taxonomy with optional per-item distributions."""

    name: str
    label_order: tuple[str, ...]
    assignments: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    default_label: str | None = None

    def __post_init__(self) -> None:
        self.name = validate_tree_id(self.name)
        labels = tuple(self.label_order)
        if not labels:
            raise InvalidIdentityError("taxonomy label_order must be non-empty")
        if len(labels) != len(set(labels)):
            raise InvalidIdentityError("taxonomy labels must be unique")
        object.__setattr__(self, "label_order", labels)
        if self.default_label is None:
            object.__setattr__(self, "default_label", labels[0])
        elif self.default_label not in labels:
            raise InvalidIdentityError("default_label must be in label_order")

    def transform(self, documents: Sequence[Document]) -> np.ndarray:
        rows: list[list[float]] = []
        labels = self.label_order
        for doc in documents:
            assigned = self.assignments.get(doc.item_id)
            if assigned is None:
                vector = [
                    1.0 if label == self.default_label else 0.0 for label in labels
                ]
            else:
                if set(assigned) != set(labels) or len(assigned) != len(labels):
                    raise InvalidIdentityError(
                        f"assignment for {doc.item_id!r} must cover label_order exactly"
                    )
                vector = [float(assigned[label]) for label in labels]
                total = sum(vector)
                if total <= 0 or not np.isfinite(total):
                    raise InvalidIdentityError(
                        f"assignment for {doc.item_id!r} must have positive mass"
                    )
                vector = [value / total for value in vector]
            rows.append(vector)
        return np.asarray(rows, dtype=np.float64)
