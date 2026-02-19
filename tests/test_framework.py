from __future__ import annotations

import networkx as nx

from pathfinding_framework import PathfindingFramework


def build_line_graph() -> nx.Graph:
    graph = nx.Graph()
    names = ["A", "B", "C", "D", "E", "F", "G"]
    for idx, name in enumerate(names):
        graph.add_node(idx, name=name)
    for left, right in zip(range(len(names) - 1), range(1, len(names))):
        graph.add_edge(left, right, relation="road")
    return graph


def build_two_cluster_graph() -> nx.Graph:
    graph = nx.Graph()
    names = ["A", "B", "C", "D", "E", "F"]
    for idx, name in enumerate(names):
        graph.add_node(idx, name=name)

    cluster_one = [(0, 1), (1, 2), (0, 2)]
    cluster_two = [(3, 4), (4, 5), (3, 5)]
    bridge = [(2, 3)]

    for edge in cluster_one + cluster_two:
        graph.add_edge(*edge, relation="intra")
    for edge in bridge:
        graph.add_edge(*edge, relation="bridge")
    return graph


def test_end_to_end_solve_returns_valid_path() -> None:
    graph = build_line_graph()
    framework = PathfindingFramework(verbose=False)
    framework.initialize_graph(graph, method="greedy", preprocess=False)

    result = framework.solve(
        source="A",
        target="G",
        graph=graph,
        expected_distance=6,
    )

    assert result["status"] in {"success", "success_fallback"}
    assert result["final_path"][0] == "A"
    assert result["final_path"][-1] == "G"
    assert result["validation"]["is_valid"] is True
    assert result["validation"]["distance_match"] is True


def test_save_and_load_metagraph_roundtrip() -> None:
    graph = build_line_graph()
    framework = PathfindingFramework(verbose=False)
    framework.initialize_graph(graph, method="greedy", preprocess=False)

    payload = framework.graph_memory.to_dict()
    loaded = PathfindingFramework(verbose=False)
    loaded.graph_memory.load_from_dict(payload)
    mapping = loaded.graph_memory.map_entities_to_communities("A", "G")
    assert mapping["A"] is not None
    assert mapping["G"] is not None


def test_get_subgraph_relationships_when_cross_subgraph() -> None:
    graph = build_two_cluster_graph()
    framework = PathfindingFramework(verbose=False)
    framework.initialize_graph(graph, method="greedy", preprocess=False)
    mapping = framework.graph_memory.map_entities_to_communities("A", "F")

    source_subgraph = mapping["A"]
    target_subgraph = mapping["F"]
    if source_subgraph != target_subgraph:
        relationships = framework.graph_memory.get_subgraph_relationships(
            source_subgraph,
            target_subgraph,
        )
        assert relationships
        assert "Subgraph" in relationships[0]
