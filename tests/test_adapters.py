"""Phase 4 adapter normalization tests."""

from __future__ import annotations

from pathlib import Path

from orchard import OrchardBuilder
from orchard.adapters import (
    documents_from_csv,
    documents_from_directory,
    documents_from_json,
    documents_from_jsonl,
    documents_from_records,
)
from orchard.adapters.loaders import load_documents
from orchard.backends.tfidf import TfidfEmbeddingBackend


def test_adapters_normalize_and_feed_build(tmp_path: Path) -> None:
    records = [
        {"item_id": "a", "text": "calendar reminder scheduling"},
        {"item_id": "b", "text": "email messaging summary"},
        {"item_id": "c", "text": "create task proposal"},
        {"item_id": "d", "text": "search budget documents"},
    ]
    json_path = tmp_path / "docs.json"
    json_path.write_text(__import__("json").dumps(records), encoding="utf-8")
    jsonl_path = tmp_path / "docs.jsonl"
    jsonl_path.write_text(
        "\n".join(__import__("json").dumps(row) for row in records) + "\n",
        encoding="utf-8",
    )
    csv_path = tmp_path / "docs.csv"
    csv_path.write_text(
        "item_id,text,title\n"
        + "\n".join(f"{r['item_id']},{r['text']},{r['item_id']}" for r in records)
        + "\n",
        encoding="utf-8",
    )
    dir_path = tmp_path / "notes"
    dir_path.mkdir()
    for row in records:
        (dir_path / f"{row['item_id']}.txt").write_text(row["text"], encoding="utf-8")

    assert len(documents_from_records(records)) == 4
    assert len(documents_from_json(json_path)) == 4
    assert len(documents_from_jsonl(jsonl_path)) == 4
    assert len(documents_from_csv(csv_path)) == 4
    assert len(documents_from_directory(dir_path)) == 4

    docs = load_documents(jsonl_path)
    orchard = OrchardBuilder(
        taxonomies=[],
        embedding_backend=TfidfEmbeddingBackend(),
    ).build(docs)
    assert orchard.tree("semantic").leaf_count == 4
