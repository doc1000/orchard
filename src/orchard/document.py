"""Canonical document records shared across Orchard trees."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

from orchard.exceptions import InvalidIdentityError
from orchard.identity import generate_item_id
from orchard.schemas import DOCUMENT_SCHEMA_VERSION


@dataclass(frozen=True, slots=True)
class Document:
    """One corpus item; equals one leaf in every Orchard tree for that corpus.

    Only ``text`` is required. Missing ``item_id`` values are generated stably
    from ``text``.
    """

    text: str
    item_id: str = ""
    title: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)
    source: str | None = None
    schema_version: str = DOCUMENT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        text = self.text if isinstance(self.text, str) else str(self.text)
        object.__setattr__(self, "text", text)
        item_id = self.item_id.strip() if self.item_id else ""
        if not item_id:
            item_id = generate_item_id(text)
        if not item_id:
            raise InvalidIdentityError("item_id must be non-empty")
        object.__setattr__(self, "item_id", item_id)
        object.__setattr__(self, "title", self.title if self.title is not None else "")
        metadata = dict(self.metadata) if self.metadata is not None else {}
        object.__setattr__(self, "metadata", metadata)
        if self.schema_version != DOCUMENT_SCHEMA_VERSION:
            raise InvalidIdentityError(
                f"unsupported document schema_version: {self.schema_version!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return payload

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> Document:
        if "text" not in value:
            raise InvalidIdentityError("document mapping requires 'text'")
        return cls(
            text=str(value["text"]),
            item_id=str(value["item_id"]) if value.get("item_id") else "",
            title=str(value["title"]) if value.get("title") is not None else "",
            metadata=dict(value.get("metadata") or {}),
            source=None if value.get("source") is None else str(value["source"]),
            schema_version=str(
                value.get("schema_version", DOCUMENT_SCHEMA_VERSION)
            ),
        )
