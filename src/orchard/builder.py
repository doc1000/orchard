"""OrchardBuilder: construct multi-tree Orchards from documents."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from orchard.backends.similarity import (
    cosine_matrix,
    jensen_shannon_matrix,
    linkage_from_similarity,
    validate_similarity_matrix,
)
from orchard.backends.tfidf import TfidfEmbeddingBackend
from orchard.document import Document
from orchard.exceptions import InvalidIdentityError
from orchard.identity import ensure_unique_item_ids, validate_tree_id
from orchard.orchard import Orchard
from orchard.taxonomy import Taxonomy, default_taxonomies
from orchard.tree import Tree

_UNSET: Any = object()


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
    - default / ``None`` taxonomies → packaged Domain + Function trees.
    - ``taxonomies == []`` → one ``semantic`` tree (default TF-IDF cosine path).
    - explicit non-empty ``taxonomies`` → one named tree per taxonomy.

    No fused Domain+Function+semantic default tree.
    Public construction verb is ``build`` (not ``fit_transform``).
    """

    taxonomies: Any = field(default=_UNSET)
    embedding_backend: Any | None = None
    linkage_method: str = "average"
    semantic_signed_cosine: bool = False
    taxonomy_similarity: str = "jensen_shannon"
    include_semantic_with_taxonomies: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.taxonomy_similarity not in {"jensen_shannon", "cosine"}:
            raise InvalidIdentityError(
                "taxonomy_similarity must be 'jensen_shannon' or 'cosine'"
            )
        if self.taxonomies is _UNSET or self.taxonomies is None:
            self.taxonomies = default_taxonomies()
        else:
            self.taxonomies = list(self.taxonomies)
        names = [validate_tree_id(taxonomy.name) for taxonomy in self.taxonomies]
        if len(names) != len(set(names)):
            raise InvalidIdentityError("taxonomy names must be unique tree ids")
        if self.embedding_backend is None:
            self.embedding_backend = TfidfEmbeddingBackend()

    def get_params(self) -> dict[str, Any]:
        """Inspectable builder configuration (sklearn-style)."""
        return {
            "taxonomies": [taxonomy.name for taxonomy in self.taxonomies],
            "embedding_backend": type(self.embedding_backend).__name__,
            "linkage_method": self.linkage_method,
            "semantic_signed_cosine": self.semantic_signed_cosine,
            "taxonomy_similarity": self.taxonomy_similarity,
            "include_semantic_with_taxonomies": self.include_semantic_with_taxonomies,
            "metadata": dict(self.metadata),
        }

    def build(
        self,
        documents: Sequence[Document | str | Mapping[str, Any]],
    ) -> Orchard:
        docs = normalize_documents(documents)
        trees: dict[str, Tree] = {}

        if not self.taxonomies:
            trees["semantic"] = self._build_semantic_tree(docs)
        else:
            for taxonomy in self.taxonomies:
                trees[taxonomy.name] = self._build_taxonomy_tree(docs, taxonomy)
            if self.include_semantic_with_taxonomies:
                trees["semantic"] = self._build_semantic_tree(docs)

        return Orchard.from_trees(
            documents=docs,
            trees=trees,
            metadata={
                "builder": self.get_params(),
                **self.metadata,
            },
        )

    def _build_semantic_tree(self, documents: Sequence[Document]) -> Tree:
        texts = [doc.text for doc in documents]
        features = np.asarray(self.embedding_backend.encode(texts), dtype=np.float64)
        if features.shape[0] != len(documents):
            raise InvalidIdentityError("embedding backend row count must match documents")
        similarity = cosine_matrix(features, signed=self.semantic_signed_cosine)
        return self._tree_from_similarity(
            similarity,
            documents,
            tree_id="semantic",
        )

    def _build_taxonomy_tree(
        self,
        documents: Sequence[Document],
        taxonomy: Taxonomy,
    ) -> Tree:
        distributions = np.asarray(taxonomy.transform(documents), dtype=np.float64)
        if distributions.shape[0] != len(documents):
            raise InvalidIdentityError("taxonomy transform row count must match documents")
        if self.taxonomy_similarity == "jensen_shannon":
            similarity = jensen_shannon_matrix(distributions)
        else:
            similarity = cosine_matrix(distributions, signed=False)
        return self._tree_from_similarity(
            similarity,
            documents,
            tree_id=taxonomy.name,
        )

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
