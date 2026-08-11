"""Dynamic cuts, Steiner subtrees, packing, and walk helpers.

Extracted from tool-tree-demo adapters/dynamic_tree_cutter.py,
adapters/linkage_cluster_packer.py, and notebook walk helpers.
Leaf identity field renamed tool_id → item_id.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.cluster.hierarchy import cophenet, fcluster
from scipy.spatial.distance import squareform
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

from orchard.schemas import (
    DYNAMIC_CUT_SCHEMA_VERSION,
    DYNAMIC_CUTTER_SCHEMA_VERSION,
    PACKED_VIEW_SCHEMA_VERSION,
    PACKER_ADAPTER_SCHEMA_VERSION,
)
from orchard.tree import Tree, default_node_label

_OPTIMIZER_BY_NAME = {
    "calinski_harabasz_score": calinski_harabasz_score,
    "silhouette_score": silhouette_score,
    "davies_bouldin_score": davies_bouldin_score,
}


def _groups(labels: np.ndarray, item_ids: Sequence[str]) -> list[tuple[str, ...]]:
    grouped: dict[int, list[str]] = defaultdict(list)
    for index, label in enumerate(labels):
        grouped[int(label)].append(item_ids[index])
    order = {item_id: index for index, item_id in enumerate(item_ids)}
    return sorted(
        (tuple(sorted(values)) for values in grouped.values()),
        key=lambda values: min(order[value] for value in values),
    )


def _score_width(
    visible_width: int,
    child_count: int,
    *,
    min_width: int,
    max_width: int,
    target_width: int,
) -> float | None:
    if visible_width <= 1:
        return None
    score = abs(visible_width - target_width)
    if visible_width < min_width:
        score += 3 * (min_width - visible_width)
    if visible_width > max_width:
        score += 3 * (visible_width - max_width)
    if child_count == 0:
        score += 1000
    return float(score)


def _resolve_optimizer(
    cut_optimizer: str | Callable[..., float] | None,
) -> Callable[..., float]:
    if cut_optimizer is None:
        return calinski_harabasz_score
    if callable(cut_optimizer):
        return cut_optimizer
    try:
        return _OPTIMIZER_BY_NAME[cut_optimizer]
    except KeyError as exc:
        known = ", ".join(sorted(_OPTIMIZER_BY_NAME))
        raise ValueError(
            f"unknown cut_optimizer {cut_optimizer!r}; expected callable or one of: {known}"
        ) from exc


def linkage_for_canonical_subtree(
    tree: Mapping[str, Any],
    cluster_node_id: str,
    *,
    linkage: Mapping[str, Any] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Build a SciPy Z matrix and local item list for one canonical branch."""
    nodes = tree["nodes"]
    cluster = nodes[cluster_node_id]
    if cluster["kind"] != "internal":
        raise ValueError("subtree root must be an internal canonical node")

    linkage_payload = linkage if linkage is not None else tree.get("linkage")
    if linkage_payload is None:
        raise ValueError("linkage is required on the tree or as an argument")

    full_item_ids = list(
        linkage_payload.get("item_ids") or linkage_payload.get("tool_ids")
    )
    member_set = set(cluster["descendant_item_ids"])
    local_item_ids = [item_id for item_id in full_item_ids if item_id in member_set]
    leaf_count = len(local_item_ids)
    if leaf_count < 2:
        raise ValueError("need at least 2 leaves")

    leaf_local = {item_id: index for index, item_id in enumerate(local_item_ids)}

    internals: list[Mapping[str, Any]] = []
    stack = [cluster_node_id]
    while stack:
        node_id = stack.pop()
        node = nodes[node_id]
        if node["kind"] == "leaf":
            continue
        internals.append(node)
        stack.extend(node["children"])
    internals.sort(key=lambda node: node["linkage_row"])
    if len(internals) != leaf_count - 1:
        raise ValueError("subtree internal count does not match leaf count")

    node_local: dict[str, int] = {}
    for node_id, node in nodes.items():
        if node["kind"] == "leaf" and node["item_id"] in leaf_local:
            node_local[node_id] = leaf_local[node["item_id"]]

    z_matrix = np.empty((leaf_count - 1, 4), dtype=np.float64)
    for index, node in enumerate(internals):
        left, right = node["children"]
        z_matrix[index, 0] = node_local[left]
        z_matrix[index, 1] = node_local[right]
        z_matrix[index, 2] = node["linkage_distance"]
        z_matrix[index, 3] = node["descendant_count"]
        node_local[node["node_id"]] = leaf_count + index

    return z_matrix, local_item_ids


def get_optimal_cut_labels(
    z_matrix: np.ndarray,
    matrix: np.ndarray | None = None,
    *,
    metric: str | Callable[..., float] | None = None,
    polarity: int = 1,
    min_clusters: int = 2,
    max_clusters: int = 17,
    precomputed: bool | None = None,
) -> tuple[np.ndarray, int]:
    """Return flat labels and cluster count for the best scoring maxclust cut."""
    if polarity not in {-1, 1}:
        raise ValueError("polarity must be 1 or -1")
    z_matrix = np.asarray(z_matrix, dtype=np.float64)
    if z_matrix.ndim != 2 or z_matrix.shape[1] != 4 or z_matrix.shape[0] < 1:
        raise ValueError("z_matrix must have shape (n_leaves - 1, 4)")
    leaf_count = int(z_matrix.shape[0] + 1)
    upper = min(max_clusters, leaf_count - 1)
    lower = max(2, min_clusters)
    if lower > upper:
        raise ValueError("cluster search range is empty for this linkage")

    score_kwargs: dict[str, Any] = {}
    if matrix is None:
        feature_matrix = squareform(cophenet(z_matrix))
        optimizer = _resolve_optimizer(metric or "silhouette_score")
        use_precomputed = True if precomputed is None else precomputed
        if use_precomputed and optimizer is not silhouette_score:
            raise ValueError(
                "matrix-less optimal cuts use cophenetic distances; "
                "pass feature_matrix or use silhouette_score"
            )
    else:
        feature_matrix = np.asarray(matrix)
        if feature_matrix.shape[0] != leaf_count:
            raise ValueError("matrix rows must match linkage leaf count")
        optimizer = _resolve_optimizer(metric or "calinski_harabasz_score")
        use_precomputed = False if precomputed is None else precomputed
    if use_precomputed:
        score_kwargs["metric"] = "precomputed"

    best_score = float("-inf")
    best_k = lower
    for cluster_count in range(lower, upper + 1):
        labels = fcluster(z_matrix, t=cluster_count, criterion="maxclust")
        if len({int(label) for label in labels}) < 2:
            continue
        try:
            score = float(optimizer(feature_matrix, labels, **score_kwargs)) * polarity
        except ValueError:
            continue
        if score > best_score:
            best_score = score
            best_k = cluster_count
    if best_score == float("-inf"):
        raise ValueError("no valid optimal cut found in cluster search range")
    return fcluster(z_matrix, t=best_k, criterion="maxclust"), best_k


def _subset_feature_matrix(
    feature_matrix: np.ndarray | None,
    full_item_ids: Sequence[str],
    local_item_ids: Sequence[str],
) -> np.ndarray | None:
    if feature_matrix is None:
        return None
    index = {item_id: position for position, item_id in enumerate(full_item_ids)}
    rows = [index[item_id] for item_id in local_item_ids]
    return np.asarray(feature_matrix)[rows]


def build_dynamic_cut(
    tree: Mapping[str, Any] | Tree,
    linkage: Mapping[str, Any] | None = None,
    *,
    top_criterion: str,
    cluster_count: int | None = None,
    distance_threshold: float | None = None,
    distance_threshold_normalized: float | None = None,
    min_width: int = 3,
    max_width: int = 7,
    target_width: int | None = 5,
    max_depth: int = 4,
    threshold_steps: int = 32,
    cut_optimizer: str | Callable[..., float] | None = None,
    cut_polarity: int = 1,
    feature_matrix: np.ndarray | None = None,
) -> dict[str, Any]:
    """Build a nested canonical cut with configurable top cut, width, and depth."""
    if isinstance(tree, Tree):
        canonical = tree.to_canonical_dict()
        linkage_payload = linkage or tree.linkage_payload()
    else:
        canonical = tree
        if linkage is None:
            raise ValueError("linkage is required when tree is a mapping")
        linkage_payload = linkage

    if top_criterion not in {"cluster_count", "distance_threshold", "optimal"}:
        raise ValueError(
            "top_criterion must be cluster_count, distance_threshold, or optimal"
        )
    if min_width < 1 or max_width < min_width:
        raise ValueError("width bounds are invalid")
    if top_criterion != "optimal":
        if target_width is None:
            raise ValueError("target_width is required unless top_criterion is optimal")
        if not min_width <= target_width <= max_width:
            raise ValueError("target_width must fall within width bounds")
    elif target_width is not None and not min_width <= target_width <= max_width:
        raise ValueError("target_width must fall within width bounds")
    if max_depth < 0 or threshold_steps < 1:
        raise ValueError("max_depth and threshold_steps are invalid")
    if cut_polarity not in {-1, 1}:
        raise ValueError("cut_polarity must be 1 or -1")

    z_matrix = np.asarray(linkage_payload["z_matrix"], dtype=np.float64)
    item_ids = tuple(
        linkage_payload.get("item_ids") or linkage_payload.get("tool_ids")
    )
    if z_matrix.shape != (len(item_ids) - 1, 4):
        raise ValueError("linkage shape does not match item IDs")
    root_members = canonical["nodes"][canonical["root_node_id"]]["descendant_item_ids"]
    if set(item_ids) != set(root_members):
        raise ValueError("linkage and canonical tree leaf memberships differ")
    if feature_matrix is not None and np.asarray(feature_matrix).shape[0] != len(
        item_ids
    ):
        raise ValueError("feature_matrix rows must match linkage item IDs")

    nodes = canonical["nodes"]
    node_by_membership = {
        tuple(node["descendant_item_ids"]): node_id for node_id, node in nodes.items()
    }
    leaf_node_by_item = {
        node["item_id"]: node_id
        for node_id, node in nodes.items()
        if node["kind"] == "leaf"
    }
    heights = np.unique(z_matrix[:, 2])
    if len(heights) > threshold_steps:
        positions = np.linspace(0, len(heights) - 1, threshold_steps).round().astype(int)
        candidate_thresholds = np.unique(heights[positions])
    else:
        candidate_thresholds = heights
    labels_by_threshold = {
        float(threshold): fcluster(z_matrix, t=float(threshold), criterion="distance")
        for threshold in candidate_thresholds
    }

    stop_width = max_width if target_width is None else target_width
    resolved_optimizer = None
    optimizer_name = None
    if top_criterion == "optimal":
        selected_optimizer: str | Callable[..., float]
        if cut_optimizer is None:
            selected_optimizer = (
                "calinski_harabasz_score"
                if feature_matrix is not None
                else "silhouette_score"
            )
        else:
            selected_optimizer = cut_optimizer
        resolved_optimizer = _resolve_optimizer(selected_optimizer)
        if isinstance(selected_optimizer, str):
            optimizer_name = selected_optimizer
        else:
            optimizer_name = getattr(selected_optimizer, "__name__", "callable")

    root_height = float(nodes[canonical["root_node_id"]]["linkage_distance"])
    resolved_threshold = None
    if top_criterion == "cluster_count":
        if cluster_count is None or not 1 <= cluster_count <= len(item_ids):
            raise ValueError("cluster_count is required and must be in leaf range")
        top_labels = fcluster(z_matrix, t=cluster_count, criterion="maxclust")
    elif top_criterion == "distance_threshold":
        if (distance_threshold is None) == (distance_threshold_normalized is None):
            raise ValueError(
                "provide exactly one of distance_threshold or "
                "distance_threshold_normalized"
            )
        if distance_threshold_normalized is not None:
            if not 0 <= distance_threshold_normalized <= 1:
                raise ValueError("normalized distance threshold must be in [0, 1]")
            resolved_threshold = root_height * distance_threshold_normalized
        else:
            resolved_threshold = float(distance_threshold)
        if resolved_threshold < 0:
            raise ValueError("distance threshold must be non-negative")
        top_labels = fcluster(z_matrix, t=resolved_threshold, criterion="distance")
    else:
        if cluster_count is not None:
            raise ValueError("cluster_count cannot be combined with optimal cuts")
        if distance_threshold is not None or distance_threshold_normalized is not None:
            raise ValueError("distance thresholds cannot be combined with optimal cuts")
        k_hi = min(max_width, len(item_ids) - 1)
        k_lo = max(2, min_width)
        if k_lo > k_hi:
            raise ValueError("width bounds leave no valid optimal cluster count")
        top_labels, cluster_count = get_optimal_cut_labels(
            z_matrix,
            None if feature_matrix is None else np.asarray(feature_matrix),
            metric=resolved_optimizer,
            polarity=cut_polarity,
            min_clusters=k_lo,
            max_clusters=k_hi,
            precomputed=feature_matrix is None,
        )

    def canonical_id(membership: tuple[str, ...]) -> str:
        try:
            return node_by_membership[membership]
        except KeyError as exc:
            raise ValueError("flat cut produced a non-canonical membership") from exc

    top_memberships = _groups(top_labels, item_ids)
    top_partition_ids = [canonical_id(values) for values in top_memberships]
    warnings: list[str] = []

    def best_subcut(node_id: str) -> tuple[float, list[tuple[str, ...]]] | None:
        node = nodes[node_id]
        parent_height = float(node["linkage_distance"])
        membership = set(node["descendant_item_ids"])
        best: tuple[float, float, list[tuple[str, ...]]] | None = None
        for threshold in candidate_thresholds:
            threshold = float(threshold)
            if threshold >= parent_height:
                continue
            labels = labels_by_threshold[threshold]
            memberships = [
                values
                for values in _groups(labels, item_ids)
                if set(values).issubset(membership)
            ]
            if sum(len(values) for values in memberships) != len(membership):
                raise ValueError("subcut does not cover its canonical parent")
            child_count = sum(len(values) > 1 for values in memberships)
            assert target_width is not None
            score = _score_width(
                len(memberships),
                child_count,
                min_width=min_width,
                max_width=max_width,
                target_width=target_width,
            )
            if score is None:
                continue
            candidate = (score, -threshold, memberships)
            if best is None or candidate[:2] < best[:2]:
                best = candidate
        return None if best is None else (-best[1], best[2])

    def best_optimal_subcut(
        node_id: str,
    ) -> tuple[int, list[tuple[str, ...]]] | None:
        node = nodes[node_id]
        if node["kind"] != "internal":
            return None
        leaf_count = int(node["descendant_count"])
        k_hi = min(max_width, leaf_count - 1)
        k_lo = max(2, min_width)
        if k_lo > k_hi:
            return None
        z_sub, local_item_ids = linkage_for_canonical_subtree(
            canonical, node_id, linkage=linkage_payload
        )
        local_matrix = _subset_feature_matrix(feature_matrix, item_ids, local_item_ids)
        labels, chosen_k = get_optimal_cut_labels(
            z_sub,
            local_matrix,
            metric=resolved_optimizer,
            polarity=cut_polarity,
            min_clusters=k_lo,
            max_clusters=k_hi,
            precomputed=local_matrix is None,
        )
        memberships = _groups(labels, local_item_ids)
        if sum(len(values) for values in memberships) != leaf_count:
            raise ValueError("optimal subcut does not cover its canonical parent")
        return chosen_k, memberships

    def build_cluster(node_id: str, depth: int, cut_value: float | None) -> dict[str, Any]:
        node = nodes[node_id]
        result = {
            "canonical_node_id": node_id,
            "membership_hash": node["membership_hash"],
            "descendant_count": node["descendant_count"],
            "depth": depth,
            "cut_value": cut_value,
            "direct_leaf_node_ids": [],
            "children": [],
        }
        if node["kind"] == "leaf":
            result["direct_leaf_node_ids"] = [node_id]
            return result
        if node["descendant_count"] <= stop_width or depth >= max_depth:
            result["direct_leaf_node_ids"] = [
                leaf_node_by_item[value] for value in node["descendant_item_ids"]
            ]
            if depth >= max_depth and node["descendant_count"] > max_width:
                warnings.append(
                    f"{node_id} retains {node['descendant_count']} leaves at "
                    f"max_depth={max_depth}"
                )
            return result
        if top_criterion == "optimal":
            selected_optimal = best_optimal_subcut(node_id)
            if selected_optimal is None:
                result["direct_leaf_node_ids"] = [
                    leaf_node_by_item[value] for value in node["descendant_item_ids"]
                ]
                warnings.append(f"{node_id} has no valid optimal subcut")
                return result
            _, memberships = selected_optimal
            child_cut_value = None
        else:
            selected = best_subcut(node_id)
            if selected is None:
                result["direct_leaf_node_ids"] = [
                    leaf_node_by_item[value] for value in node["descendant_item_ids"]
                ]
                warnings.append(f"{node_id} has no lower canonical distance cut")
                return result
            threshold, memberships = selected
            child_cut_value = threshold
        for values in memberships:
            child_id = canonical_id(values)
            if len(values) == 1:
                result["direct_leaf_node_ids"].append(child_id)
            else:
                result["children"].append(
                    build_cluster(child_id, depth + 1, child_cut_value)
                )
        return result

    root_id = canonical["root_node_id"]
    root = {
        "canonical_node_id": root_id,
        "membership_hash": nodes[root_id]["membership_hash"],
        "descendant_count": nodes[root_id]["descendant_count"],
        "depth": 0,
        "cut_value": resolved_threshold,
        "direct_leaf_node_ids": [],
        "children": [],
    }
    for membership in top_memberships:
        node_id = canonical_id(membership)
        if len(membership) == 1:
            root["direct_leaf_node_ids"].append(node_id)
        else:
            root["children"].append(build_cluster(node_id, 1, resolved_threshold))

    return {
        "schema_version": DYNAMIC_CUT_SCHEMA_VERSION,
        "adapter_schema_version": DYNAMIC_CUTTER_SCHEMA_VERSION,
        "top_criterion": top_criterion,
        "cluster_count_requested": cluster_count,
        "distance_threshold_requested": distance_threshold,
        "distance_threshold_normalized_requested": distance_threshold_normalized,
        "distance_threshold_resolved": resolved_threshold,
        "cut_optimizer": optimizer_name,
        "cut_polarity": cut_polarity if top_criterion == "optimal" else None,
        "top_partition_node_ids": top_partition_ids,
        "top_partition_count": len(top_partition_ids),
        "width": {
            "minimum": min_width,
            "maximum": max_width,
            "target": target_width,
            "stop": stop_width,
        },
        "max_depth": max_depth,
        "threshold_steps": threshold_steps,
        "candidate_distance_thresholds": [
            float(value) for value in candidate_thresholds
        ],
        "warnings": warnings,
        "root": root,
    }


def validate_dynamic_cut(cut: Mapping[str, Any], tree: Mapping[str, Any] | Tree) -> None:
    """Verify canonical mapping and exact leaf coverage for a generated cut."""
    canonical = tree.to_canonical_dict() if isinstance(tree, Tree) else tree
    if cut.get("schema_version") != DYNAMIC_CUT_SCHEMA_VERSION:
        raise ValueError("unsupported dynamic cut schema")
    nodes = canonical["nodes"]
    expected = nodes[canonical["root_node_id"]]["descendant_item_ids"]

    partition_members = []
    for node_id in cut["top_partition_node_ids"]:
        if node_id not in nodes:
            raise ValueError("top partition references a non-canonical node")
        partition_members.extend(nodes[node_id]["descendant_item_ids"])
    if sorted(partition_members) != expected or len(partition_members) != len(
        set(partition_members)
    ):
        raise ValueError("top cut is not an exact canonical leaf partition")

    seen_leaves: list[str] = []
    stack = [cut["root"]]
    while stack:
        item = stack.pop()
        source = nodes.get(item["canonical_node_id"])
        if source is None:
            raise ValueError("nested cut references a non-canonical node")
        if (
            item["membership_hash"] != source["membership_hash"]
            or item["descendant_count"] != source["descendant_count"]
        ):
            raise ValueError("nested cut canonical membership mismatch")
        local_members = []
        for leaf_node_id in item["direct_leaf_node_ids"]:
            leaf = nodes.get(leaf_node_id)
            if leaf is None or leaf["kind"] != "leaf":
                raise ValueError("direct leaf reference is not canonical")
            local_members.append(leaf["item_id"])
            seen_leaves.append(leaf["item_id"])
        for child in item["children"]:
            child_source = nodes.get(child["canonical_node_id"])
            if child_source is None:
                raise ValueError("child cut reference is not canonical")
            local_members.extend(child_source["descendant_item_ids"])
        if sorted(local_members) != source["descendant_item_ids"]:
            raise ValueError("nested cut node does not retain exact membership")
        stack.extend(item["children"])
    if sorted(seen_leaves) != expected or len(seen_leaves) != len(set(seen_leaves)):
        raise ValueError("nested cut does not retain every expected leaf exactly once")


def pack_canonical_tree(
    tree: Mapping[str, Any] | Tree,
    *,
    target_exposed_clusters: int,
    max_depth: int,
    max_fanout: int = 2,
) -> dict[str, Any]:
    """Return a bounded presentation tree whose frontier references canonical nodes."""
    canonical = tree.to_canonical_dict() if isinstance(tree, Tree) else tree
    if target_exposed_clusters < 1:
        raise ValueError("target_exposed_clusters must be positive")
    if max_depth < 0:
        raise ValueError("max_depth must be non-negative")
    if max_fanout != 2:
        raise ValueError("binary canonical trees require max_fanout=2")

    nodes = canonical["nodes"]
    root_id = canonical["root_node_id"]
    frontier: dict[str, int] = {root_id: 0}
    expanded: set[str] = set()

    while len(frontier) < target_exposed_clusters:
        candidates = [
            (depth, -int(nodes[node_id]["descendant_count"]), node_id)
            for node_id, depth in frontier.items()
            if nodes[node_id]["kind"] == "internal" and depth < max_depth
        ]
        if not candidates:
            break
        _, _, selected = min(candidates)
        depth = frontier.pop(selected)
        children = nodes[selected]["children"]
        if len(children) > max_fanout:
            raise ValueError("canonical child count exceeds configured max_fanout")
        expanded.add(selected)
        for child_id in children:
            frontier[child_id] = depth + 1

    def build(node_id: str, depth: int) -> dict[str, Any]:
        source = nodes[node_id]
        payload = {
            "canonical_node_id": node_id,
            "membership_hash": source["membership_hash"],
            "descendant_count": source["descendant_count"],
            "depth": depth,
            "children": [],
            "exposed_cluster": node_id in frontier,
        }
        if node_id in expanded:
            payload["children"] = [
                build(child_id, depth + 1) for child_id in source["children"]
            ]
        return payload

    packed_root = build(root_id, 0)
    return {
        "schema_version": PACKED_VIEW_SCHEMA_VERSION,
        "adapter_schema_version": PACKER_ADAPTER_SCHEMA_VERSION,
        "target_interpretation": (
            "desired count of canonical frontier clusters; the actual count can "
            "be lower when max_depth prevents further canonical splits"
        ),
        "target_exposed_clusters": target_exposed_clusters,
        "actual_exposed_clusters": len(frontier),
        "max_depth": max_depth,
        "max_fanout": max_fanout,
        "root": packed_root,
        "frontier_node_ids": sorted(frontier),
    }


def cut_partition_item_ids(
    cut: Mapping[str, Any],
    tree: Mapping[str, Any] | Tree,
) -> list[list[str]]:
    """Return the top-cut partition as lists of item IDs (exact leaf cover)."""
    canonical = tree.to_canonical_dict() if isinstance(tree, Tree) else tree
    nodes = canonical["nodes"]
    return [
        list(nodes[node_id]["descendant_item_ids"])
        for node_id in cut["top_partition_node_ids"]
    ]


def walk_cut_json(
    cut: Mapping[str, Any],
    tree: Tree,
    *,
    label_set: str | None = None,
    max_depth: int | None = None,
) -> dict[str, Any]:
    """Serialize a cut to a nested label tree for viewers/agents."""

    def walk(node: Mapping[str, Any]) -> dict[str, Any]:
        node_id = node["canonical_node_id"]
        depth = int(node["depth"])
        payload = {
            "canonical_node_id": node_id,
            "label": default_node_label(tree, node_id, label_set=label_set),
            "descendant_count": node["descendant_count"],
            "depth": depth,
            "children": [],
        }
        if max_depth is not None and depth >= max_depth:
            return payload
        children = []
        for leaf_id in node.get("direct_leaf_node_ids", []):
            children.append(
                {
                    "canonical_node_id": leaf_id,
                    "label": default_node_label(tree, leaf_id, label_set=label_set),
                    "descendant_count": 1,
                    "depth": depth + 1,
                    "children": [],
                }
            )
        for child in node.get("children", []):
            children.append(walk(child))
        payload["children"] = children
        return payload

    return walk(cut["root"])
