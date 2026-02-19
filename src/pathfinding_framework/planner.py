from __future__ import annotations

from collections import deque
from typing import Dict, List, Set


def shortest_subgraph_route(
    meta_graph: Dict[str, List[str]],
    source_subgraph: str,
    target_subgraph: str,
) -> List[str]:
    """Return shortest subgraph-hop route in the metagraph."""

    source = str(source_subgraph)
    target = str(target_subgraph)
    if source == target:
        return [source]
    if not meta_graph:
        return []

    adjacency: Dict[str, Set[str]] = {}
    for node, neighbors in meta_graph.items():
        node_key = str(node)
        adjacency.setdefault(node_key, set())
        for neighbor in neighbors:
            neighbor_key = str(neighbor)
            adjacency[node_key].add(neighbor_key)
            adjacency.setdefault(neighbor_key, set()).add(node_key)

    if source not in adjacency or target not in adjacency:
        return []

    parents: Dict[str, str | None] = {source: None}
    queue: deque[str] = deque([source])

    while queue:
        current = queue.popleft()
        if current == target:
            break
        for neighbor in adjacency.get(current, set()):
            if neighbor in parents:
                continue
            parents[neighbor] = current
            queue.append(neighbor)

    if target not in parents:
        return []

    route: List[str] = []
    cursor: str | None = target
    while cursor is not None:
        route.append(cursor)
        cursor = parents.get(cursor)
    route.reverse()
    return route
