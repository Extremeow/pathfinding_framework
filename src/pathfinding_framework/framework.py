from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import networkx as nx

from .combiner import run_combiner_agent
from .llm_client import DEFAULT_MODEL, OpenAIChatClient
from .memory import GraphMemory
from .partitioning import DEFAULT_RESOLUTION, partition_graph
from .planner import shortest_subgraph_route
from .solver import run_solver_agent
from .utils import build_name_graph, sort_subgraph_ids
from .validation import validate_path


MASTER_SYSTEM_PROMPT = """You are the Master agent for hierarchical pathfinding.

You must solve by calling tools, not by inventing paths.

Available tools:
- graph_extract: inspect metagraph topology.
- dispatch_solver: run a local LLM solver in one subgraph.
- get_subgraph_relationships: get portal transitions from last dispatched subgraph to a target subgraph.
- dispatch_combiner: combine worker paths into one final path.

Execution policy:
1) First call graph_extract on the source subgraph.
2) Dispatch solver in current subgraph.
3) If target subgraph not yet reached, use get_subgraph_relationships to select entry node for next subgraph.
4) Continue until target is reached.
5) Call dispatch_combiner exactly once at the end.

Return concise final text after combiner is done.
"""


def _safe_json_load(text: str) -> Dict[str, Any]:
    try:
        value = json.loads(text)
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


def _usage_delta(before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, Any]:
    keys = {"calls", "input_tokens", "output_tokens", "total_tokens", "cost"}
    delta: Dict[str, Any] = {}
    for key in keys:
        delta[key] = after.get(key, 0) - before.get(key, 0)
    return delta


class PathfindingFramework:
    """
    LLM-based Master/Solver/Combiner pathfinding framework.
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        max_turns: int = 30,
        verbose: bool = True,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.max_turns = max_turns
        self.verbose = verbose

        self.graph_memory = GraphMemory()
        self.worker_responses: List[Dict[str, Any]] = []
        self._name_graph: Optional[nx.Graph] = None
        self._llm_client: Optional[OpenAIChatClient] = None

        self.last_resolution: Optional[float] = None
        self.last_partition_method: Optional[str] = None
        self.last_solve_result: Optional[Dict[str, Any]] = None

        # Master runtime state
        self._dispatch_count = 0
        self._last_dispatched_subgraph: Optional[str] = None
        self._last_portal_nodes: List[str] = []
        self._last_combiner_result: Optional[Dict[str, Any]] = None

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message)

    def _ensure_client(self) -> OpenAIChatClient:
        if self._llm_client is None:
            self._llm_client = OpenAIChatClient(api_key=self.api_key, model=self.model)
        return self._llm_client

    def initialize_graph(
        self,
        graph: nx.Graph,
        resolution: float = DEFAULT_RESOLUTION,
        seed: int = 42,
        method: str = "auto",
        preprocess: bool = True,
        save_path: Optional[str | Path] = None,
    ) -> Dict[str, Any]:
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
        self._dispatch_count = 0
        self._last_dispatched_subgraph = None
        self._last_portal_nodes = []
        self._last_combiner_result = None

    def _auto_priority_subgraph(self, current_subgraph: str, target_subgraph: str) -> Optional[str]:
        if current_subgraph == target_subgraph:
            return target_subgraph
        route = shortest_subgraph_route(
            self.graph_memory.meta_graph,
            source_subgraph=current_subgraph,
            target_subgraph=target_subgraph,
        )
        if len(route) >= 2:
            return route[1]
        return None

    def _tool_schemas(self) -> List[Dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "graph_extract",
                    "description": "Extract metagraph slice around a subgraph for route planning.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "subgraph_id": {"type": "string"},
                            "hop": {"type": "integer", "default": 2},
                        },
                        "required": ["subgraph_id"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "dispatch_solver",
                    "description": "Run local solver for one subgraph.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "subgraph_id": {"type": "string"},
                            "source": {"type": "string"},
                            "priority_subgraph": {"type": "string"},
                        },
                        "required": ["subgraph_id", "source"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_subgraph_relationships",
                    "description": (
                        "Get portal relationships from the last dispatched subgraph "
                        "to a target subgraph."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "target_subgraph_id": {"type": "string"},
                        },
                        "required": ["target_subgraph_id"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "dispatch_combiner",
                    "description": "Combine all worker paths into final path JSON.",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
            },
        ]

    def _run_graph_extract_tool(self, function_args: Dict[str, Any]) -> str:
        subgraph_id = str(function_args.get("subgraph_id"))
        hop = int(function_args.get("hop", 2))
        result = self.graph_memory.graph_extract(subgraph_id=subgraph_id, hop=hop)
        return json.dumps(result, ensure_ascii=False)

    def _run_get_relationships_tool(self, function_args: Dict[str, Any]) -> str:
        target_subgraph = str(function_args.get("target_subgraph_id"))
        if self._last_dispatched_subgraph is None:
            return json.dumps(
                {
                    "error": "No previous dispatch_solver call found.",
                    "relationships": [],
                },
                ensure_ascii=False,
            )

        current_subgraph = self._last_dispatched_subgraph
        allowed_portals = self._last_portal_nodes if self._last_portal_nodes else None
        relationships = self.graph_memory.get_subgraph_relationships(
            current_subgraph=current_subgraph,
            target_subgraph=target_subgraph,
            allowed_portals=allowed_portals,
        )
        links = self.graph_memory.candidate_portal_links(current_subgraph, target_subgraph)
        if allowed_portals:
            allowed = set(allowed_portals)
            links = [link for link in links if link.portal_node in allowed]
        entry_nodes = sorted({link.entry_node for link in links})

        return json.dumps(
            {
                "current_subgraph": current_subgraph,
                "target_subgraph": target_subgraph,
                "relationships": relationships,
                "entry_nodes": entry_nodes,
            },
            ensure_ascii=False,
        )

    def _run_dispatch_solver_tool(
        self,
        *,
        function_args: Dict[str, Any],
        target: str,
        target_subgraph: str,
    ) -> str:
        subgraph_id = str(function_args.get("subgraph_id"))
        source = str(function_args.get("source"))
        requested_priority = function_args.get("priority_subgraph")
        priority_subgraph = (
            str(requested_priority)
            if requested_priority is not None
            else self._auto_priority_subgraph(subgraph_id, target_subgraph)
        )

        self._dispatch_count += 1
        client = self._ensure_client()
        solver_result = run_solver_agent(
            client=client,
            memory=self.graph_memory,
            subgraph_id=subgraph_id,
            source=source,
            target=target,
            priority_subgraph=priority_subgraph,
        )

        record = {
            "dispatch_id": self._dispatch_count,
            "subgraph_id": subgraph_id,
            "source": source,
            "target": target,
            "priority_subgraph": priority_subgraph,
            "result": solver_result,
        }
        self.worker_responses.append(record)

        self._last_dispatched_subgraph = subgraph_id
        self._last_portal_nodes = [str(x) for x in solver_result.get("portal_nodes", [])]

        connected_subgraphs = sort_subgraph_ids(
            solver_result.get("subgraph_connections", {}).keys()
        )
        if not connected_subgraphs:
            connected_subgraphs = self.graph_memory.subgraph_neighbors(subgraph_id)

        reached_target_subgraph = (
            str(subgraph_id) == str(target_subgraph)
            and str(solver_result.get("status", "")).lower() == "reached"
            and bool(solver_result.get("paths_found"))
            and str(solver_result["paths_found"][0][-1]) == str(target)
        )

        tool_result = {
            "status": solver_result.get("status", "failed"),
            "current_subgraph": subgraph_id,
            "priority_subgraph": priority_subgraph,
            "connected_subgraphs": connected_subgraphs,
            "portal_nodes": self._last_portal_nodes,
            "reached_target_subgraph": reached_target_subgraph,
            "assignment": solver_result.get("assignment"),
            "error": solver_result.get("error"),
        }
        return json.dumps(tool_result, ensure_ascii=False)

    def _run_dispatch_combiner_tool(self, *, source: str, target: str) -> str:
        client = self._ensure_client()
        self._last_combiner_result = run_combiner_agent(
            client=client,
            worker_records=self.worker_responses,
            source=source,
            target=target,
        )
        return json.dumps(self._last_combiner_result, ensure_ascii=False)

    def _execute_tool(
        self,
        *,
        function_name: str,
        function_args: Dict[str, Any],
        source: str,
        target: str,
        target_subgraph: str,
    ) -> str:
        if function_name == "graph_extract":
            return self._run_graph_extract_tool(function_args)
        if function_name == "get_subgraph_relationships":
            return self._run_get_relationships_tool(function_args)
        if function_name == "dispatch_solver":
            return self._run_dispatch_solver_tool(
                function_args=function_args,
                target=target,
                target_subgraph=target_subgraph,
            )
        if function_name == "dispatch_combiner":
            return self._run_dispatch_combiner_tool(source=source, target=target)
        return json.dumps({"error": f"Unknown tool: {function_name}"}, ensure_ascii=False)

    def solve(
        self,
        source: str,
        target: str,
        source_subgraph: Optional[str] = None,
        target_subgraph: Optional[str] = None,
        graph: Optional[nx.Graph] = None,
        expected_distance: Optional[int] = None,
    ) -> Dict[str, Any]:
        solve_start = time.perf_counter()
        self.reset_state()
        client = self._ensure_client()
        usage_before = client.usage_totals.to_dict()

        if graph is not None:
            self._name_graph = build_name_graph(graph)

        if not self.graph_memory.initialized:
            raise RuntimeError("Graph memory is empty. Run initialize_graph or load_graph_memory first.")

        source_name = str(source)
        target_name = str(target)
        mapping = self.graph_memory.map_entities_to_communities(source_name, target_name)
        source_subgraph = str(source_subgraph or mapping[source_name])
        target_subgraph = str(target_subgraph or mapping[target_name])

        self._log("=" * 80)
        self._log("SOLVE (LLM Master/Solver/Combiner)")
        self._log("=" * 80)
        self._log(f"Source: {source_name} (Subgraph {source_subgraph})")
        self._log(f"Target: {target_name} (Subgraph {target_subgraph})")

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": MASTER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Find the shortest path from '{source_name}' to '{target_name}'.\n"
                    f"Source subgraph: {source_subgraph}\n"
                    f"Target subgraph: {target_subgraph}\n"
                    "Start by calling graph_extract on the source subgraph."
                ),
            },
        ]

        final_output_text = ""
        for turn in range(1, self.max_turns + 1):
            response = client.chat_completion(
                messages=messages,
                tools=self._tool_schemas(),
                tool_choice="auto",
                temperature=0.0,
            )
            message = response.choices[0].message

            assistant_message: Dict[str, Any] = {
                "role": "assistant",
                "content": message.content or "",
            }
            if message.tool_calls:
                assistant_message["tool_calls"] = []
                for tool_call in message.tool_calls:
                    assistant_message["tool_calls"].append(
                        {
                            "id": tool_call.id,
                            "type": "function",
                            "function": {
                                "name": tool_call.function.name,
                                "arguments": tool_call.function.arguments or "{}",
                            },
                        }
                    )
            messages.append(assistant_message)

            if not message.tool_calls:
                final_output_text = message.content or ""
                self._log(f"[Turn {turn}] Master message: {final_output_text[:180]}")
                continue

            for tool_call in message.tool_calls:
                function_name = tool_call.function.name
                try:
                    function_args = json.loads(tool_call.function.arguments or "{}")
                except Exception:
                    function_args = {}
                tool_output = self._execute_tool(
                    function_name=function_name,
                    function_args=function_args,
                    source=source_name,
                    target=target_name,
                    target_subgraph=target_subgraph,
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": function_name,
                        "content": tool_output,
                    }
                )
                if self.verbose:
                    self._log(f"[Turn {turn}] Tool: {function_name} -> {tool_output[:220]}")

                if function_name == "dispatch_combiner":
                    final_output_text = tool_output
                    break
            if self._last_combiner_result is not None:
                break

        combiner_result = self._last_combiner_result or _safe_json_load(final_output_text)
        final_path_field = combiner_result.get("final_path", [])
        final_path = [str(node) for node in final_path_field] if isinstance(final_path_field, list) else []

        status = str(combiner_result.get("status", "failed")).lower()
        if not final_path:
            status = "failed"

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

        usage_after = client.usage_totals.to_dict()
        token_usage = _usage_delta(usage_before, usage_after)

        result = {
            "final_path": final_path,
            "status": status,
            "source": source_name,
            "target": target_name,
            "source_subgraph": source_subgraph,
            "target_subgraph": target_subgraph,
            "worker_count": len(self.worker_responses),
            "worker_responses": self.worker_responses,
            "combiner_result": combiner_result,
            "strategy": "llm_master_solver_combiner",
            "validation": validation,
            "final_output": final_output_text,
            "solve_time_seconds": time.perf_counter() - solve_start,
            "token_usage": token_usage,
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
