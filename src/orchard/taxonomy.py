"""Taxonomy protocols, cue-based models, and default Domain/Function assets.

Default Domain/Function taxonomies load packaged ModernBERT+logistic heads
when ``orchard[taxonomy-ml]`` (or an injected encoder) is present.
``transform()`` is then ``modernbert_logistic``. Cue is the explicit offline
fallback, not the neural default.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from orchard.document import Document
from orchard.exceptions import InvalidIdentityError
from orchard.identity import validate_tree_id

TOKEN_RE = re.compile(r"[a-z0-9]+")
TAXONOMY_DEFINITION_SCHEMA = "orchard_taxonomy_definition_v1"


@runtime_checkable
class Taxonomy(Protocol):
    """Semantic classification structure used as a per-tree build signal."""

    @property
    def name(self) -> str: ...

    @property
    def label_order(self) -> tuple[str, ...]: ...

    def transform(self, documents: Sequence[Document]) -> np.ndarray: ...


@dataclass(frozen=True, slots=True)
class TaxonomyLabel:
    label_id: str
    name: str
    definition: str
    cues: tuple[str, ...]


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


def _document_tokens(document: Document) -> set[str]:
    text = f"{document.title} {document.text}".casefold()
    return set(TOKEN_RE.findall(text))


def _document_title_text(document: Document) -> str:
    return f"{document.title} {document.text}".strip()


def taxonomy_ml_extra_available() -> bool:
    from orchard.backends.modernbert import (
        taxonomy_ml_extra_available as extra_available,
    )

    return extra_available()


def missing_taxonomy_ml_error():
    from orchard.backends.modernbert import missing_taxonomy_ml_error as missing_error

    return missing_error()


@dataclass
class TaxonomyModel:
    """Inspectable taxonomy with cue transform and optional classifier head."""

    name: str
    label_order: tuple[str, ...]
    labels: dict[str, TaxonomyLabel]
    taxonomy_version: str = "orchard_taxonomy_v1"
    provenance: str = ""
    taxonomy_transform: str = "cue"
    vectorizer: TfidfVectorizer | None = field(default=None, repr=False)
    classifier: LogisticRegression | None = field(default=None, repr=False)
    feature_encoder: Any | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.name = validate_tree_id(self.name)
        ordered = tuple(self.label_order)
        if not ordered:
            raise InvalidIdentityError("label_order must be non-empty")
        if set(ordered) != set(self.labels):
            raise InvalidIdentityError("labels must match label_order exactly")
        self.label_order = ordered

    @classmethod
    def from_definition(cls, path: str | Path | Mapping[str, Any]) -> TaxonomyModel:
        if isinstance(path, Mapping):
            payload = dict(path)
        else:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("schema_version") != TAXONOMY_DEFINITION_SCHEMA:
            raise InvalidIdentityError(
                f"unsupported taxonomy schema: {payload.get('schema_version')!r}"
            )
        labels = {
            row["label_id"]: TaxonomyLabel(
                label_id=row["label_id"],
                name=row["name"],
                definition=row["definition"],
                cues=tuple(row.get("cues") or ()),
            )
            for row in payload["labels"]
        }
        return cls(
            name=str(payload["name"]),
            label_order=tuple(payload["label_order"]),
            labels=labels,
            taxonomy_version=str(payload.get("taxonomy_version") or "orchard_taxonomy_v1"),
            provenance=str(payload.get("provenance") or ""),
        )

    @classmethod
    def load_default(cls, name: str) -> TaxonomyModel:
        """Load a packaged default taxonomy definition by name (cue-capable JSON)."""
        name = validate_tree_id(name)
        resource = resources.files("orchard.assets.taxonomies").joinpath(f"{name}.json")
        if not resource.is_file():
            raise InvalidIdentityError(f"no packaged default taxonomy named {name!r}")
        return cls.from_definition(json.loads(resource.read_text(encoding="utf-8")))

    def to_definition(self) -> dict[str, Any]:
        return {
            "schema_version": TAXONOMY_DEFINITION_SCHEMA,
            "name": self.name,
            "taxonomy_version": self.taxonomy_version,
            "structure": "flat",
            "provenance": self.provenance,
            "label_order": list(self.label_order),
            "labels": [
                {
                    "label_id": label.label_id,
                    "name": label.name,
                    "definition": label.definition,
                    "cues": list(label.cues),
                }
                for label_id in self.label_order
                for label in (self.labels[label_id],)
            ],
        }

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.to_definition(), ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
        return target

    @classmethod
    def load(cls, path: str | Path) -> TaxonomyModel:
        return cls.from_definition(path)

    def fit(
        self,
        documents: Sequence[Document],
        labels_by_item_id: Mapping[str, str],
    ) -> TaxonomyModel:
        """Fit a lightweight TF-IDF + logistic head on labeled documents."""
        texts: list[str] = []
        y: list[str] = []
        for doc in documents:
            label = labels_by_item_id.get(doc.item_id)
            if label is None:
                continue
            if label not in self.labels:
                raise InvalidIdentityError(f"unknown label {label!r} for {doc.item_id}")
            texts.append(_document_title_text(doc))
            y.append(label)
        if len(set(y)) < 2:
            raise InvalidIdentityError("fit requires at least two distinct labels")
        vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b[\w][\w-]*\b")
        features = vectorizer.fit_transform(texts)
        classifier = LogisticRegression(max_iter=1000)
        classifier.fit(features, y)
        self.vectorizer = vectorizer
        self.classifier = classifier
        self.feature_encoder = None
        self.taxonomy_transform = "cue"
        return self

    def attach_packaged_head(self, encoder: Any | None = None) -> TaxonomyModel:
        """Load the packaged ModernBERT logistic head for this taxonomy name."""
        from orchard.backends.taxonomy_heads import load_packaged_classifier

        self.classifier = load_packaged_classifier(self.name, self.label_order)
        self.vectorizer = None
        self.feature_encoder = encoder
        self.taxonomy_transform = "modernbert_logistic"
        return self

    def load_head(self, path: str | Path, encoder: Any | None = None) -> TaxonomyModel:
        """Replace the logistic head from a portable ``npz`` (does not re-train)."""
        from orchard.backends.taxonomy_heads import load_head as load_head_arrays

        self.classifier = load_head_arrays(path, self.label_order)
        self.vectorizer = None
        if encoder is not None:
            self.feature_encoder = encoder
        self.taxonomy_transform = "modernbert_logistic"
        return self

    def save_head(self, path: str | Path) -> Path:
        """Write the current logistic head as portable arrays."""
        if self.classifier is None:
            raise InvalidIdentityError("no classifier head to save")
        from orchard.backends.taxonomy_heads import save_head as save_head_arrays

        return save_head_arrays(
            path,
            self.classifier.coef_,
            self.classifier.intercept_,
            self.classifier.classes_.tolist(),
        )

    def transform(self, documents: Sequence[Document]) -> np.ndarray:
        if self.taxonomy_transform == "modernbert_logistic":
            if self.classifier is None or self.feature_encoder is None:
                raise InvalidIdentityError(
                    "modernbert_logistic transform requires a loaded head and "
                    "feature encoder; do not silently use cue"
                )
            texts = [_document_title_text(doc) for doc in documents]
            features = np.asarray(self.feature_encoder.encode(texts), dtype=np.float64)
            if features.shape[0] != len(documents):
                raise InvalidIdentityError(
                    "taxonomy encoder row count must match documents"
                )
            from orchard.backends.taxonomy_heads import predict_proba_rows

            return predict_proba_rows(self.classifier, features, self.label_order)
        if self.classifier is not None and self.vectorizer is not None:
            texts = [_document_title_text(doc) for doc in documents]
            features = self.vectorizer.transform(texts)
            probabilities = self.classifier.predict_proba(features)
            class_index = {label: index for index, label in enumerate(self.classifier.classes_)}
            rows = []
            for row in probabilities:
                vector = [
                    float(row[class_index[label]]) if label in class_index else 0.0
                    for label in self.label_order
                ]
                total = sum(vector)
                if total <= 0:
                    vector = [1.0 / len(self.label_order)] * len(self.label_order)
                else:
                    vector = [value / total for value in vector]
                rows.append(vector)
            return np.asarray(rows, dtype=np.float64)
        return self._cue_transform(documents)

    def _cue_transform(self, documents: Sequence[Document]) -> np.ndarray:
        rows: list[list[float]] = []
        n_labels = len(self.label_order)
        for doc in documents:
            tokens = _document_tokens(doc)
            scores = [
                float(len(tokens & set(self.labels[label_id].cues)))
                for label_id in self.label_order
            ]
            if sum(scores) <= 0:
                rows.append([1.0 / n_labels] * n_labels)
                continue
            arr = np.asarray(scores, dtype=np.float64) + 0.05
            rows.append((arr / arr.sum()).tolist())
        return np.asarray(rows, dtype=np.float64)


def _load_product_default(
    name: str,
    *,
    allow_offline_fallback: bool = False,
    classifier_backend: Any | None = None,
) -> TaxonomyModel:
    from orchard.backends.modernbert import ModernBERTFeatureBackend

    model = TaxonomyModel.load_default(name)
    if classifier_backend is not None:
        return model.attach_packaged_head(classifier_backend)
    if taxonomy_ml_extra_available():
        return model.attach_packaged_head(ModernBERTFeatureBackend())
    if allow_offline_fallback:
        model.taxonomy_transform = "cue"
        return model
    raise missing_taxonomy_ml_error()


class DomainTaxonomy:
    """Convenience loader for the packaged Domain taxonomy."""

    @staticmethod
    def load_default(
        *,
        allow_offline_fallback: bool = False,
        classifier_backend: Any | None = None,
    ) -> TaxonomyModel:
        return _load_product_default(
            "domain",
            allow_offline_fallback=allow_offline_fallback,
            classifier_backend=classifier_backend,
        )


class FunctionTaxonomy:
    """Convenience loader for the packaged Function taxonomy."""

    @staticmethod
    def load_default(
        *,
        allow_offline_fallback: bool = False,
        classifier_backend: Any | None = None,
    ) -> TaxonomyModel:
        return _load_product_default(
            "function",
            allow_offline_fallback=allow_offline_fallback,
            classifier_backend=classifier_backend,
        )


def default_taxonomies(
    *,
    allow_offline_fallback: bool = False,
    classifier_backend: Any | None = None,
) -> list[TaxonomyModel]:
    """Return replaceable Domain + Function defaults (no AppWorld runtime)."""
    return [
        DomainTaxonomy.load_default(
            allow_offline_fallback=allow_offline_fallback,
            classifier_backend=classifier_backend,
        ),
        FunctionTaxonomy.load_default(
            allow_offline_fallback=allow_offline_fallback,
            classifier_backend=classifier_backend,
        ),
    ]
