from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple

import networkx as nx


def load_problem_set(path: str | Path) -> List[Dict[str, Any]]:
    dataset_path = Path(path)
    with dataset_path.open("rb") as handle:
        return pickle.load(handle)


def resolve_problem(problem: Dict[str, Any]) -> Tuple[nx.Graph, str, str, int | None]:
    graph = problem["graph"]
    source_id = problem["source"]
    target_id = problem["target"]
    source_name = str(graph.nodes[source_id].get("name", source_id))
    target_name = str(graph.nodes[target_id].get("name", target_id))
    expected_distance = problem.get("exact_answer")
    return graph, source_name, target_name, expected_distance
