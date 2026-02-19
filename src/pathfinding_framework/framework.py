from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import networkx as nx

from .combiner import combine_worker_paths
from .memory import GraphMemory
from .partitioning import DEFAULT_RESOLUTION, partition_graph
from .planner import shortest_subgraph_route
from .solver import dispatch_solver
from .utils import build_name_graph
from .validation import validate_path


class PathfindingFramework:
    """
    Deterministic hierarchical pathfinding framework.

    Workflow:
    1. Partition graph into subgraphs.
    2. Build metagraph memory.
    3. Plan shortest subgraph route.
    4. Dispatch local solver across route.
    5. Combine local segments and validate final path.
    """

    def __init__(self, verbose: bool = True) -> None:
        self.verbose = verbose
        self.graph_memory = GraphMemory()
        self.worker_responses: List[Dict[str, Any]] = []
        self._name_graph: Optional[nx.Graph] = None
        self.last_resolution: Optional[float] = None
        self.last_partition_method: Optional[str] = None
        self.last_solve_result: Optional[Dict[str, Any]] = None

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message)

    def initialize_graph(
        self,
        graph: nx.Graph,
        resolution: float = DEFAULT_RESOLUTION,
        seed: int = 42,
        method: str = "auto",
        preprocess: bool = True,
        save_path: Optional[str | Path] = None,
    ) -> Dict[str, Any]:
        """
        Initialize graph memory from a raw NetworkX graph.
        """

        self._log("=" * 80)
        self._log("INITIALIZING GRAPH MEMORY")
        self._log("=" * 80)

        name_graph = build_name_graph(graph)
        original_nodes = name_graph.number_of_nodes()
        original_edges = name_graph.number_of_edges()

        if preprocess and name_graph.number_of_nodes() > 0 and not nx.is_connected(name_graph):
            components = list(nx.connected_components(name_graph))
            largest_component = max(components, key=len)
            name_graph = name_graph.subgraph(largest_component).copy()
            self._log(
                "Preprocess: extracted largest connected component "
                f"({name_graph.number_of_nodes()} nodes, {name_graph.number_of_edges()} edges)."
            )

        partition_result = partition_graph(
            graph=name_graph,
            resolution=resolution,
            seed=seed,
            method=method,
        )
        self.graph_memory.load_from_dict(partition_result)
        self._name_graph = name_graph
        self.last_resolution = resolution
        self.last_partition_method = str(partition_result.get("partition_method", method))

        if save_path is not None:
            self.save_graph_memory(save_path)

        summary = {
            "original_nodes": original_nodes,
            "original_edges": original_edges,
            "active_nodes": name_graph.number_of_nodes(),
            "active_edges": name_graph.number_of_edges(),
            "num_subgraphs": partition_result["num_subgraphs"],
            "modularity": partition_result["modularity"],
            "partition_method": partition_result.get("partition_method", method),
            "resolution": resolution,
        }

        self._log(
            f"Partitioned into {summary['num_subgraphs']} subgraphs "
            f"(modularity={summary['modularity']:.4f}, method={summary['partition_method']})."
        )
        self._log("=" * 80)
        return summary

    def save_graph_memory(self, path: str | Path) -> None:
        self.graph_memory.save_json(path)
        self._log(f"Saved graph memory to: {Path(path)}")

    def load_graph_memory(self, path: str | Path) -> None:
        self.graph_memory.load_json(path)
        self._log(f"Loaded graph memory from: {Path(path)}")

    def reset_state(self) -> None:
        self.worker_responses = []

    def _global_shortest_path(self, source: str, target: str) -> List[str]:
        if self._name_graph is None:
            return []
        if source not in self._name_graph or target not in self._name_graph:
            return []
        try:
            return list(nx.shortest_path(self._name_graph, source, target))
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []

    def solve(
        self,
        source: str,
        target: str,
        source_subgraph: Optional[str] = None,
        target_subgraph: Optional[str] = None,
        graph: Optional[nx.Graph] = None,
        expected_distance: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Solve source -> target path with hierarchical deterministic routing.
        """

        solve_start = time.perf_counter()
        self.reset_state()

        if graph is not None:
            self._name_graph = build_name_graph(graph)

        if not self.graph_memory.initialized:
            raise RuntimeError("Graph memory is empty. Run initialize_graph or load_graph_memory first.")

        source_name = str(source)
        target_name = str(target)
        mapping = self.graph_memory.map_entities_to_communities(source_name, target_name)
        source_subgraph = str(source_subgraph or mapping[source_name])
        target_subgraph = str(target_subgraph or mapping[target_name])

        route = shortest_subgraph_route(
            self.graph_memory.meta_graph,
            source_subgraph=source_subgraph,
            target_subgraph=target_subgraph,
        )

        if not route:
            route = [source_subgraph] if source_subgraph == target_subgraph else []

        self._log("=" * 80)
        self._log("SOLVE")
        self._log("=" * 80)
        self._log(f"Source: {source_name} (Subgraph {source_subgraph})")
        self._log(f"Target: {target_name} (Subgraph {target_subgraph})")
        self._log(f"Subgraph route: {route if route else 'unreachable'}")

        status = "failed"
        strategy = "hierarchical"
        current_source = source_name

        if route:
            for idx, current_subgraph in enumerate(route):
                is_last = idx == len(route) - 1
                priority_subgraph = current_subgraph if is_last else route[idx + 1]

                record = dispatch_solver(
                    memory=self.graph_memory,
                    dispatch_id=idx + 1,
                    subgraph_id=current_subgraph,
                    source=current_source,
                    target=target_name,
                    priority_subgraph=priority_subgraph,
                ).to_dict()
                self.worker_responses.append(record)

                payload = record.get("result", {})
                step_status = payload.get("status")
                if self.verbose:
                    self._log(
                        f"[Dispatch {record['dispatch_id']}] subgraph={current_subgraph} "
                        f"status={step_status}"
                    )

                if step_status == "failed":
                    status = "failed"
                    break

                if is_last:
                    status = "success" if step_status == "reached" else "failed"
                    break

                transition = payload.get("selected_transition")
                if not transition:
                    status = "failed"
                    break
                current_source = str(transition["linked_node"])

        combiner_result = combine_worker_paths(self.worker_responses, source_name, target_name)
        final_path = combiner_result.get("final_path", [])
        if status != "success" or combiner_result.get("status") != "success":
            fallback_path = self._global_shortest_path(source_name, target_name)
            if fallback_path:
                status = "success_fallback"
                strategy = "global_shortest_path_fallback"
                final_path = fallback_path
                combiner_result = {
                    "final_path": fallback_path,
                    "status": "success",
                    "confidence": 1.0,
                    "explanation": "Fallback to global shortest path in named graph.",
                }

        validation = None
        if self._name_graph is not None and final_path:
            validation = validate_path(
                final_path,
                self._name_graph,
                source_name,
                target_name,
                expected_distance=expected_distance,
            )
            if not validation.get("is_valid", False):
                status = "failed_validation"

        result = {
            "final_path": final_path,
            "status": status,
            "source": source_name,
            "target": target_name,
            "source_subgraph": source_subgraph,
            "target_subgraph": target_subgraph,
            "subgraph_route": route,
            "worker_count": len(self.worker_responses),
            "worker_responses": self.worker_responses,
            "combiner_result": combiner_result,
            "strategy": strategy,
            "validation": validation,
            "solve_time_seconds": time.perf_counter() - solve_start,
            "token_usage": {
                "calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "cost": 0.0,
            },
        }

        self.last_solve_result = result
        return result

    def get_statistics(self) -> Dict[str, Any]:
        total_dispatches = len(self.worker_responses)
        total_paths = 0
        total_portals = 0
        for record in self.worker_responses:
            payload = record.get("result", {})
            total_paths += len(payload.get("paths_found", []))
            total_portals += len(payload.get("portal_nodes", []))

        return {
            "total_dispatches": total_dispatches,
            "total_paths": total_paths,
            "total_portals": total_portals,
            "avg_paths_per_dispatch": (total_paths / total_dispatches) if total_dispatches else 0.0,
        }
