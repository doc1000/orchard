"""Generic reference fixtures shipped with Orchard (no AppWorld data)."""

from __future__ import annotations

import json
from importlib import resources
from typing import Any

import numpy as np

from orchard.document import Document
from orchard.identity import LinkageIdentity, assign_canonical_ids


def _read_text(name: str) -> str:
    return resources.files("orchard.fixtures").joinpath(name).read_text(encoding="utf-8")


def load_documents() -> list[Document]:
    """Load the tiny generic document corpus fixture."""
    rows = json.loads(_read_text("documents.json"))
    return [Document.from_mapping(row) for row in rows]


def load_sample_linkage() -> dict[str, Any]:
    """Load sample linkage metadata and Z matrix for the document fixture."""
    payload = json.loads(_read_text("sample_linkage.json"))
    payload["linkage"] = np.asarray(payload["linkage"], dtype=float)
    return payload


def load_sample_linkage_identity() -> LinkageIdentity:
    """Assign canonical IDs for the sample linkage fixture."""
    payload = load_sample_linkage()
    return assign_canonical_ids(payload["linkage"], payload["leaf_ids"])
