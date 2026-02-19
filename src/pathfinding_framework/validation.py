from __future__ import annotations

from typing import Any, Dict, List, Optional

import networkx as nx


def validate_path(
    path: List[str],
    graph: nx.Graph,
    source: str,
    target: str,
    expected_distance: Optional[int] = None,
) -> Dict[str, Any]:
    errors: List[str] = []

    if not path:
        return {
            "is_valid": False,
            "path_length": 0,
            "errors": ["Path is empty."],
            "source_match": False,
            "target_match": False,
            "all_nodes_exist": False,
            "all_edges_exist": False,
            "distance_match": expected_distance is None,
            "expected_distance": expected_distance,
        }

    source_match = path[0] == source
    target_match = path[-1] == target
    if not source_match:
        errors.append(f"Path starts with '{path[0]}', expected '{source}'.")
    if not target_match:
        errors.append(f"Path ends with '{path[-1]}', expected '{target}'.")

    missing_nodes = [node for node in path if node not in graph]
    all_nodes_exist = len(missing_nodes) == 0
    if missing_nodes:
        errors.append(f"Missing nodes in graph: {missing_nodes}.")

    missing_edges: List[tuple[str, str]] = []
    for idx in range(len(path) - 1):
        u = path[idx]
        v = path[idx + 1]
        if not graph.has_edge(u, v):
            missing_edges.append((u, v))

    all_edges_exist = len(missing_edges) == 0
    if missing_edges:
        errors.append(f"Missing edges in graph: {missing_edges}.")

    path_length = len(path) - 1
    if expected_distance is None:
        distance_match = True
    else:
        distance_match = path_length == expected_distance

    is_valid = source_match and target_match and all_nodes_exist and all_edges_exist

    return {
        "is_valid": is_valid,
        "path_length": path_length,
        "errors": errors,
        "source_match": source_match,
        "target_match": target_match,
        "all_nodes_exist": all_nodes_exist,
        "all_edges_exist": all_edges_exist,
        "distance_match": distance_match,
        "expected_distance": expected_distance,
    }
