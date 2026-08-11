"""Phase 4: local docs site opens without a build server."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "docs" / "site"

REQUIRED_PAGES = [
    "index.html",
    "architecture.html",
    "documents.html",
    "taxonomies.html",
    "building-trees.html",
    "multiple-trees.html",
    "cuts-and-views.html",
    "labels.html",
    "persistence.html",
    "visualization.html",
    "adapters.html",
    "extending.html",
]


def test_docs_site_pages_and_sidebar_exist() -> None:
    assert (SITE / "assets" / "site.css").is_file()
    assert (SITE / "assets" / "site.js").is_file()
    for name in REQUIRED_PAGES:
        path = SITE / name
        assert path.is_file(), name
        text = path.read_text(encoding="utf-8")
        assert 'class="sidebar"' in text
        assert "Getting Started" in text
        assert "Extending Orchard" in text
