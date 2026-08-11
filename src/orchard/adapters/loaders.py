"""Lightweight adapters: directory / JSON / JSONL / CSV / records → Document."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from orchard.builder import normalize_documents
from orchard.document import Document
from orchard.exceptions import InvalidIdentityError


def documents_from_records(
    records: Sequence[Document | str | Mapping[str, Any]],
) -> list[Document]:
    """Normalize list/dict/string records into Documents."""
    return normalize_documents(records)


def documents_from_json(path: str | Path) -> list[Document]:
    """Load a JSON array of document mappings (or a single mapping)."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, Mapping):
        return documents_from_records([payload])
    if isinstance(payload, list):
        return documents_from_records(payload)
    raise InvalidIdentityError("JSON adapter expects an object or array")


def documents_from_jsonl(path: str | Path) -> list[Document]:
    """Load one JSON object per line."""
    rows: list[Any] = []
    for line_no, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            rows.append(json.loads(stripped))
        except json.JSONDecodeError as exc:
            raise InvalidIdentityError(
                f"invalid JSONL at line {line_no}: {exc}"
            ) from exc
    return documents_from_records(rows)


def documents_from_csv(
    path: str | Path,
    *,
    text_field: str = "text",
    item_id_field: str = "item_id",
    title_field: str = "title",
    source_field: str = "source",
) -> list[Document]:
    """Load Documents from a CSV file (stdlib csv; no pandas)."""
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise InvalidIdentityError("CSV has no header row")
        rows: list[dict[str, Any]] = []
        for row in reader:
            if text_field not in row or row[text_field] is None:
                raise InvalidIdentityError(
                    f"CSV rows require a {text_field!r} column"
                )
            metadata = {
                key: value
                for key, value in row.items()
                if key
                not in {text_field, item_id_field, title_field, source_field}
                and value not in (None, "")
            }
            rows.append(
                {
                    "text": row[text_field],
                    "item_id": row.get(item_id_field) or "",
                    "title": row.get(title_field) or "",
                    "source": row.get(source_field) or None,
                    "metadata": metadata,
                }
            )
    return documents_from_records(rows)


def documents_from_directory(
    path: str | Path,
    *,
    glob: str = "*.txt",
    encoding: str = "utf-8",
) -> list[Document]:
    """Load each matching text file as one Document (stem → item_id)."""
    root = Path(path)
    if not root.is_dir():
        raise InvalidIdentityError(f"not a directory: {root}")
    files = sorted(root.glob(glob))
    if not files:
        raise InvalidIdentityError(f"no files matching {glob!r} under {root}")
    records = []
    for file_path in files:
        if not file_path.is_file():
            continue
        records.append(
            {
                "item_id": file_path.stem,
                "text": file_path.read_text(encoding=encoding),
                "title": file_path.stem,
                "source": str(file_path),
            }
        )
    return documents_from_records(records)


def load_documents(
    source: str | Path | Sequence[Document | str | Mapping[str, Any]],
) -> list[Document]:
    """Dispatch helper: path suffix / directory / in-memory records."""
    if isinstance(source, (str, Path)):
        path = Path(source)
        if path.is_dir():
            return documents_from_directory(path)
        suffix = path.suffix.lower()
        if suffix == ".json":
            return documents_from_json(path)
        if suffix == ".jsonl":
            return documents_from_jsonl(path)
        if suffix == ".csv":
            return documents_from_csv(path)
        raise InvalidIdentityError(
            f"unsupported document source suffix: {suffix!r}"
        )
    if isinstance(source, Iterable):
        return documents_from_records(list(source))
    raise InvalidIdentityError(f"unsupported document source: {type(source)!r}")
