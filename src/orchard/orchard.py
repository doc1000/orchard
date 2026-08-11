"""Multi-tree Orchard over a shared document corpus."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from orchard.document import Document
from orchard.exceptions import (
    CorpusMutationUnsupportedError,
    InvalidIdentityError,
    UnknownTreeError,
)
from orchard.identity import ensure_unique_item_ids, validate_tree_id
from orchard.schemas import (
    ARTIFACT_SCHEMA_VERSION,
    DOCUMENT_SCHEMA_VERSION,
    LABEL_SET_SCHEMA_VERSION,
)
from orchard.tree import Tree


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass
class Orchard:
    """Persistent collection of named trees sharing one document corpus."""

    documents: list[Document]
    trees: dict[str, Tree] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        ensure_unique_item_ids(doc.item_id for doc in self.documents)
        normalized: dict[str, Tree] = {}
        for name, tree in self.trees.items():
            tree_id = validate_tree_id(name)
            if tree.tree_id != tree_id:
                raise InvalidIdentityError(
                    f"tree map key {name!r} does not match tree.tree_id {tree.tree_id!r}"
                )
            self._validate_tree_corpus(tree)
            normalized[tree_id] = tree
        self.trees = normalized

    def _validate_tree_corpus(self, tree: Tree) -> None:
        corpus_ids = {doc.item_id for doc in self.documents}
        tree_ids = set(tree.item_ids)
        if tree_ids != corpus_ids:
            raise InvalidIdentityError(
                "tree leaf item_ids must exactly match the Orchard document corpus"
            )

    @classmethod
    def from_trees(
        cls,
        *,
        documents: Sequence[Document],
        trees: Mapping[str, Tree],
        metadata: Mapping[str, Any] | None = None,
    ) -> Orchard:
        return cls(
            documents=list(documents),
            trees=dict(trees),
            metadata=dict(metadata or {}),
        )

    @property
    def tree_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self.trees))

    def tree(self, name: str) -> Tree:
        try:
            return self.trees[validate_tree_id(name)]
        except KeyError as exc:
            raise UnknownTreeError(name) from exc

    def add_documents(self, *_args: Any, **_kwargs: Any) -> None:
        raise CorpusMutationUnsupportedError(
            "incremental corpus mutation is unsupported; rebuild a new Orchard"
        )

    def remove_documents(self, *_args: Any, **_kwargs: Any) -> None:
        raise CorpusMutationUnsupportedError(
            "incremental corpus mutation is unsupported; rebuild a new Orchard"
        )

    def save(self, path: str | Path) -> Path:
        """Persist as a versioned artifact directory."""
        root = Path(path)
        root.mkdir(parents=True, exist_ok=True)
        documents_payload = [doc.to_dict() for doc in self.documents]
        _write_json(root / "documents.json", documents_payload)

        tree_manifest: list[dict[str, Any]] = []
        for tree_id, tree in sorted(self.trees.items()):
            tree_dir = root / "trees" / tree_id
            _write_json(tree_dir / "canonical_tree.json", tree.to_canonical_dict())
            _write_json(tree_dir / "linkage.json", tree.linkage_payload())
            labels_dir = tree_dir / "labels"
            label_names = sorted(tree.labels)
            for label_name in label_names:
                _write_json(
                    labels_dir / f"{label_name}.json",
                    {
                        "schema_version": LABEL_SET_SCHEMA_VERSION,
                        "name": label_name,
                        "labels": tree.labels[label_name],
                    },
                )
            tree_manifest.append(
                {
                    "tree_id": tree_id,
                    "root_node_id": tree.root_node_id,
                    "leaf_count": tree.leaf_count,
                    "label_sets": label_names,
                    "active_label_set": tree.active_label_set,
                    "method": tree.method,
                }
            )

        manifest = {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "document_schema_version": DOCUMENT_SCHEMA_VERSION,
            "document_count": len(self.documents),
            "tree_ids": [entry["tree_id"] for entry in tree_manifest],
            "trees": tree_manifest,
            "metadata": self.metadata,
        }
        _write_json(root / "manifest.json", manifest)
        return root

    @classmethod
    def load(cls, path: str | Path) -> Orchard:
        root = Path(path)
        manifest = _read_json(root / "manifest.json")
        if manifest.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
            raise InvalidIdentityError(
                f"unsupported orchard artifact schema: {manifest.get('schema_version')!r}"
            )
        documents = [
            Document.from_mapping(row) for row in _read_json(root / "documents.json")
        ]
        item_ids = ensure_unique_item_ids(doc.item_id for doc in documents)
        if len(item_ids) != manifest.get("document_count"):
            raise InvalidIdentityError("document_count does not match documents.json")

        trees: dict[str, Tree] = {}
        for entry in manifest["trees"]:
            tree_id = validate_tree_id(entry["tree_id"])
            tree_dir = root / "trees" / tree_id
            canonical = _read_json(tree_dir / "canonical_tree.json")
            linkage = _read_json(tree_dir / "linkage.json")
            labels: dict[str, dict[str, str]] = {}
            for label_name in entry.get("label_sets", []):
                payload = _read_json(tree_dir / "labels" / f"{label_name}.json")
                if payload.get("schema_version") != LABEL_SET_SCHEMA_VERSION:
                    raise InvalidIdentityError(
                        f"unsupported label set schema for {label_name!r}"
                    )
                labels[label_name] = dict(payload["labels"])
            trees[tree_id] = Tree.from_persisted(
                canonical=canonical,
                linkage=linkage,
                labels=labels,
                active_label_set=entry.get("active_label_set"),
                documents=documents,
            )
            if trees[tree_id].root_node_id != entry["root_node_id"]:
                raise InvalidIdentityError(
                    f"manifest root_node_id mismatch for tree {tree_id!r}"
                )

        return cls(
            documents=documents,
            trees=trees,
            metadata=dict(manifest.get("metadata") or {}),
        )
