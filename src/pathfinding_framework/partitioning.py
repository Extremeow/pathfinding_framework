from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List

import networkx as nx

from .utils import sort_subgraph_ids

DEFAULT_RESOLUTION = 3.5


def _partition_with_leiden(
    graph: nx.Graph, resolution: float, seed: int
) -> Dict[str, str]:
    try:
        import igraph as ig  # type: ignore
        import leidenalg  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Leiden partitioning requires 'igraph' and 'leidenalg'. "
            "Install with: pip install igraph leidenalg"
        ) from exc

    node_list = list(graph.nodes())
    node_to_index = {node: idx for idx, node in enumerate(node_list)}

    ig_graph = ig.Graph()
    ig_graph.add_vertices(len(node_list))
    ig_graph.add_edges([(node_to_index[u], node_to_index[v]) for u, v in graph.edges()])

    partition = leidenalg.find_partition(
        ig_graph,
        leidenalg.RBConfigurationVertexPartition,
        resolution_parameter=resolution,
        seed=seed,
    )

    node_to_subgraph: Dict[str, str] = {}
    for subgraph_id, members in enumerate(partition):
        for member_idx in members:
            node_name = str(node_list[member_idx])
            node_to_subgraph[node_name] = str(subgraph_id)
    return node_to_subgraph


def _partition_with_greedy(graph: nx.Graph) -> Dict[str, str]:
    if graph.number_of_nodes() == 0:
        return {}
    if graph.number_of_edges() == 0:
        return {str(node): str(idx) for idx, node in enumerate(graph.nodes())}

    communities = list(nx.algorithms.community.greedy_modularity_communities(graph))
    node_to_subgraph: Dict[str, str] = {}
    for subgraph_id, members in enumerate(communities):
        for node in members:
            node_to_subgraph[str(node)] = str(subgraph_id)

    for node in graph.nodes():
        node_to_subgraph.setdefault(str(node), str(len(node_to_subgraph)))
    return node_to_subgraph


def identify_portal_nodes(
    graph: nx.Graph, node_to_subgraph: Dict[str, str]
) -> Dict[str, List[str]]:
    portal_sets: Dict[str, set[str]] = defaultdict(set)
    for source, target in graph.edges():
        source_name = str(source)
        target_name = str(target)
        source_subgraph = node_to_subgraph[source_name]
        target_subgraph = node_to_subgraph[target_name]
        if source_subgraph != target_subgraph:
            portal_sets[source_subgraph].add(source_name)
            portal_sets[target_subgraph].add(target_name)

    return {
        subgraph_id: sorted(nodes)
        for subgraph_id, nodes in portal_sets.items()
    }


def build_local_adjacencies(
    graph: nx.Graph, node_to_subgraph: Dict[str, str]
) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    local_adjacencies: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(dict)

    for node_name, subgraph_id in node_to_subgraph.items():
        local_adjacencies[subgraph_id].setdefault(node_name, [])

    for source, target, attrs in graph.edges(data=True):
        source_name = str(source)
        target_name = str(target)
        source_subgraph = node_to_subgraph[source_name]
        target_subgraph = node_to_subgraph[target_name]
        if source_subgraph != target_subgraph:
            continue

        relation = str(attrs.get("relation", "connected_to"))
        semantics = str(attrs.get("semantics", relation))
        tags = attrs.get("tags", [])
        if isinstance(tags, list):
            normalized_tags = [str(tag) for tag in tags]
        else:
            normalized_tags = [str(tags)]

        local_adjacencies[source_subgraph][source_name].append(
            {
                "target": target_name,
                "relation": relation,
                "semantics": semantics,
                "tags": normalized_tags,
            }
        )
        local_adjacencies[source_subgraph][target_name].append(
            {
                "target": source_name,
                "relation": relation,
                "semantics": semantics,
                "tags": normalized_tags,
            }
        )

    return dict(local_adjacencies)


def build_meta_graph(
    graph: nx.Graph, node_to_subgraph: Dict[str, str]
) -> Dict[str, List[str]]:
    meta_edges: Dict[str, set[str]] = defaultdict(set)
    for subgraph_id in set(node_to_subgraph.values()):
        meta_edges[subgraph_id] = set()

    for source, target in graph.edges():
        source_name = str(source)
        target_name = str(target)
        source_subgraph = node_to_subgraph[source_name]
        target_subgraph = node_to_subgraph[target_name]
        if source_subgraph == target_subgraph:
            continue
        meta_edges[source_subgraph].add(target_subgraph)
        meta_edges[target_subgraph].add(source_subgraph)

    return {
        subgraph_id: sort_subgraph_ids(neighbors)
        for subgraph_id, neighbors in meta_edges.items()
    }


def build_inter_subgraph_edges(
    graph: nx.Graph, node_to_subgraph: Dict[str, str]
) -> List[Dict[str, Any]]:
    edges: List[Dict[str, Any]] = []
    for source, target, attrs in graph.edges(data=True):
        source_name = str(source)
        target_name = str(target)
        source_subgraph = node_to_subgraph[source_name]
        target_subgraph = node_to_subgraph[target_name]
        if source_subgraph == target_subgraph:
            continue
        relation = str(attrs.get("relation", "connected_to"))
        semantics = str(attrs.get("semantics", relation))
        tags = attrs.get("tags", [])
        if isinstance(tags, list):
            normalized_tags = [str(tag) for tag in tags]
        else:
            normalized_tags = [str(tags)]

        edges.append(
            {
                "source": source_name,
                "target": target_name,
                "source_subgraph": source_subgraph,
                "target_subgraph": target_subgraph,
                "relation": relation,
                "semantics": semantics,
                "tags": normalized_tags,
            }
        )
    return edges


def _build_subgraph_lists(
    node_to_subgraph: Dict[str, str]
) -> List[List[str]]:
    subgraphs: Dict[str, List[str]] = defaultdict(list)
    for node_name, subgraph_id in node_to_subgraph.items():
        subgraphs[subgraph_id].append(node_name)

    ordered_subgraphs: List[List[str]] = []
    for subgraph_id in sort_subgraph_ids(subgraphs.keys()):
        ordered_subgraphs.append(sorted(subgraphs[subgraph_id]))
    return ordered_subgraphs


def _compute_modularity(
    graph: nx.Graph, node_to_subgraph: Dict[str, str]
) -> float:
    if graph.number_of_edges() == 0 or not node_to_subgraph:
        return 0.0
    communities_by_id: Dict[str, set[str]] = defaultdict(set)
    for node_name, subgraph_id in node_to_subgraph.items():
        communities_by_id[subgraph_id].add(node_name)
    communities: List[set[str]] = list(communities_by_id.values())
    return float(nx.algorithms.community.modularity(graph, communities))


def partition_graph(
    graph: nx.Graph,
    resolution: float = DEFAULT_RESOLUTION,
    seed: int = 42,
    method: str = "auto",
) -> Dict[str, Any]:
    """
    Partition graph into subgraphs and build metagraph artifacts.

    Args:
        graph: Name-keyed NetworkX graph.
        resolution: Leiden resolution parameter (used when Leiden is selected).
        seed: Random seed for Leiden.
        method: 'auto', 'leiden', or 'greedy'.
    """

    if method not in {"auto", "leiden", "greedy"}:
        raise ValueError("method must be one of: auto, leiden, greedy")

    if method in {"auto", "leiden"}:
        try:
            node_to_subgraph = _partition_with_leiden(graph, resolution=resolution, seed=seed)
            partition_method = "leiden"
        except ImportError:
            if method == "leiden":
                raise
            node_to_subgraph = _partition_with_greedy(graph)
            partition_method = "greedy"
    else:
        node_to_subgraph = _partition_with_greedy(graph)
        partition_method = "greedy"

    portal_nodes = identify_portal_nodes(graph, node_to_subgraph)
    local_adjacencies = build_local_adjacencies(graph, node_to_subgraph)
    meta_graph = build_meta_graph(graph, node_to_subgraph)
    inter_subgraph_edges = build_inter_subgraph_edges(graph, node_to_subgraph)
    subgraphs = _build_subgraph_lists(node_to_subgraph)
    modularity = _compute_modularity(graph, node_to_subgraph)

    return {
        "node_to_subgraph": node_to_subgraph,
        "subgraphs": subgraphs,
        "num_subgraphs": len(subgraphs),
        "modularity": modularity,
        "portal_nodes": portal_nodes,
        "local_adjacencies": local_adjacencies,
        "meta_graph": meta_graph,
        "inter_subgraph_edges": inter_subgraph_edges,
        "partition_method": partition_method,
    }


def map_nodes_to_subgraphs(
    node_to_subgraph: Dict[str, str], nodes: Iterable[str]
) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for node in nodes:
        key = str(node)
        if key not in node_to_subgraph:
            raise KeyError(f"Node '{key}' is not present in node_to_subgraph map.")
        mapping[key] = node_to_subgraph[key]
    return mapping
