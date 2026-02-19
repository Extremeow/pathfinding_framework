from __future__ import annotations

from typing import Iterable, List, Tuple

import networkx as nx


def _subgraph_sort_key(value: str) -> Tuple[int, int | str]:
    text = str(value)
    if text.isdigit():
        return (0, int(text))
    return (1, text)


def sort_subgraph_ids(values: Iterable[str]) -> List[str]:
    return sorted((str(v) for v in values), key=_subgraph_sort_key)


def build_name_graph(graph: nx.Graph) -> nx.Graph:
    """Build an undirected graph keyed by unique node names."""

    if graph.is_directed():
        base_graph: nx.Graph = nx.Graph(graph.to_undirected())
    else:
        base_graph = nx.Graph(graph)

    name_graph = nx.Graph()
    seen_names: dict[str, object] = {}

    for node_id, attrs in base_graph.nodes(data=True):
        node_name = str(attrs.get("name", node_id))
        if node_name in seen_names and seen_names[node_name] != node_id:
            raise ValueError(
                f"Duplicate node name '{node_name}' detected. "
                "Node names must be unique for this framework."
            )
        seen_names[node_name] = node_id
        name_graph.add_node(node_name, original_id=node_id)

    for source, target, attrs in base_graph.edges(data=True):
        source_name = str(base_graph.nodes[source].get("name", source))
        target_name = str(base_graph.nodes[target].get("name", target))
        relation = str(attrs.get("relation", "connected_to"))
        semantics = str(attrs.get("semantics", relation))
        tags = attrs.get("tags", [])
        if not isinstance(tags, list):
            tags = [str(tags)]
        else:
            tags = [str(tag) for tag in tags]

        name_graph.add_edge(
            source_name,
            target_name,
            relation=relation,
            semantics=semantics,
            tags=tags,
        )

    return name_graph
