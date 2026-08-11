"""Input adapters that normalize external records into Document."""

from orchard.adapters.loaders import (
    documents_from_csv,
    documents_from_directory,
    documents_from_json,
    documents_from_jsonl,
    documents_from_records,
    load_documents,
)

__all__ = [
    "documents_from_csv",
    "documents_from_directory",
    "documents_from_json",
    "documents_from_jsonl",
    "documents_from_records",
    "load_documents",
]
