"""Named label-set helpers. Contrastive generation is intentionally absent."""

from __future__ import annotations

import re
from collections import Counter
from typing import Mapping

from orchard.exceptions import InvalidIdentityError
from orchard.tree import Tree

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]*")


def import_labels(
    tree: Tree,
    mapping: Mapping[str, str],
    *,
    name: str = "imported",
    make_active: bool = True,
) -> None:
    """Import a user-provided label overlay. Never mutates linkage."""
    before = tree.linkage_fingerprint()
    tree.set_labels(name, mapping, make_active=make_active)
    if tree.linkage_fingerprint() != before:
        raise InvalidIdentityError("label import mutated linkage")


def label_intrinsic(
    tree: Tree,
    *,
    name: str = "intrinsic_v1",
    make_active: bool = True,
    max_words: int = 4,
) -> dict[str, str]:
    """Offline heuristic intrinsic labels for leaves and internal nodes.

    Leaves use title/text tokens; internals merge child labels / descendant
    texts. This preserves the phase4b1 idea of bottom-up summaries without an
    OpenAI dependency. Optional LLM backends can replace this later.
    """
    before = tree.linkage_fingerprint()
    nodes = sorted(
        tree.nodes.values(),
        key=lambda node: (int(node["descendant_count"]), node["node_id"]),
    )
    labels: dict[str, str] = {}
    for node in nodes:
        node_id = node["node_id"]
        if node["kind"] == "leaf":
            item_id = node["item_id"]
            doc = tree.documents_by_id.get(item_id)
            if doc is not None and doc.title.strip():
                text = doc.title
            elif doc is not None:
                text = doc.text
            else:
                text = item_id
            labels[node_id] = _clip_words(text, max_words)
            continue
        child_labels = [labels[child] for child in node["children"] if child in labels]
        if child_labels:
            labels[node_id] = _merge_labels(child_labels, max_words=max_words)
        else:
            member_text = " ".join(node["descendant_item_ids"])
            labels[node_id] = _clip_words(member_text, max_words)
    tree.set_labels(name, labels, make_active=make_active)
    if tree.linkage_fingerprint() != before:
        raise InvalidIdentityError("intrinsic labeling mutated linkage")
    return labels


def _clip_words(text: str, max_words: int) -> str:
    tokens = _TOKEN_RE.findall(text)
    if not tokens:
        return text.strip()[:40] or "untitled"
    return " ".join(tokens[:max_words])


def _merge_labels(child_labels: list[str], *, max_words: int) -> str:
    counts: Counter[str] = Counter()
    for label in child_labels:
        counts.update(token.casefold() for token in _TOKEN_RE.findall(label))
    if not counts:
        return " / ".join(child_labels[:2])
    top = [token for token, _ in counts.most_common(max_words)]
    return " ".join(top)
