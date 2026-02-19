from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import networkx as nx

from .models import PortalLink
from .utils import sort_subgraph_ids


class GraphMemory:
    """
    Runtime memory store for metagraph artifacts.

    Expected payload keys:
    - node_to_subgraph
    - meta_graph
    - portal_nodes
    - local_adjacencies
    - inter_subgraph_edges
    """

    def __init__(self) -> None:
        self.clear()

    def clear(self) -> None:
        self.node_to_subgraph: Dict[str, str] = {}
        self.meta_graph: Dict[str, List[str]] = {}
        self.portal_nodes: Dict[str, List[str]] = {}
        self.local_adjacencies: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
        self.inter_subgraph_edges: List[Dict[str, Any]] = []
        self._local_graph_cache: Dict[str, nx.Graph] = {}

    @property
    def initialized(self) -> bool:
        return bool(self.node_to_subgraph)

    def load_from_dict(self, payload: Dict[str, Any]) -> None:
        self.clear()

        self.node_to_subgraph = {
            str(node): str(subgraph)
            for node, subgraph in payload.get("node_to_subgraph", {}).items()
        }

        self.meta_graph = {
            str(subgraph): [str(neighbor) for neighbor in neighbors]
            for subgraph, neighbors in payload.get("meta_graph", {}).items()
        }

        self.portal_nodes = {
            str(subgraph): [str(node) for node in nodes]
            for subgraph, nodes in payload.get("portal_nodes", {}).items()
        }

        raw_local = payload.get("local_adjacencies", {})
        parsed_local: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
        for subgraph, adjacency in raw_local.items():
            normalized_subgraph = str(subgraph)
            parsed_local[normalized_subgraph] = {}
            for node, neighbors in adjacency.items():
                normalized_node = str(node)
                normalized_neighbors: List[Dict[str, Any]] = []
                for neighbor in neighbors:
                    if isinstance(neighbor, dict):
                        target = neighbor.get("target")
                        if target is None:
                            continue
                        relation = str(neighbor.get("relation", "connected_to"))
                        semantics = str(neighbor.get("semantics", relation))
                        tags = neighbor.get("tags", [])
                        if isinstance(tags, list):
                            normalized_tags = [str(tag) for tag in tags]
                        else:
                            normalized_tags = [str(tags)]
                        normalized_neighbors.append(
                            {
                                "target": str(target),
                                "relation": relation,
                                "semantics": semantics,
                                "tags": normalized_tags,
                            }
                        )
                    else:
                        normalized_neighbors.append(
                            {
                                "target": str(neighbor),
                                "relation": "connected_to",
                                "semantics": "connected_to",
                                "tags": [],
                            }
                        )
                parsed_local[normalized_subgraph][normalized_node] = normalized_neighbors
        self.local_adjacencies = parsed_local

        for node, subgraph in self.node_to_subgraph.items():
            self.local_adjacencies.setdefault(subgraph, {})
            self.local_adjacencies[subgraph].setdefault(node, [])
            self.meta_graph.setdefault(subgraph, [])
            self.portal_nodes.setdefault(subgraph, [])

        parsed_edges: List[Dict[str, Any]] = []
        for edge in payload.get("inter_subgraph_edges", []):
            if not isinstance(edge, dict):
                continue
            source = edge.get("source")
            target = edge.get("target")
            source_subgraph = edge.get("source_subgraph")
            target_subgraph = edge.get("target_subgraph")
            if None in {source, target, source_subgraph, target_subgraph}:
                continue
            relation = str(edge.get("relation", "connected_to"))
            semantics = str(edge.get("semantics", relation))
            tags = edge.get("tags", [])
            if isinstance(tags, list):
                normalized_tags = [str(tag) for tag in tags]
            else:
                normalized_tags = [str(tags)]
            parsed_edges.append(
                {
                    "source": str(source),
                    "target": str(target),
                    "source_subgraph": str(source_subgraph),
                    "target_subgraph": str(target_subgraph),
                    "relation": relation,
                    "semantics": semantics,
                    "tags": normalized_tags,
                }
            )
        self.inter_subgraph_edges = parsed_edges
        self._local_graph_cache = {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_to_subgraph": dict(self.node_to_subgraph),
            "meta_graph": {k: list(v) for k, v in self.meta_graph.items()},
            "portal_nodes": {k: list(v) for k, v in self.portal_nodes.items()},
            "local_adjacencies": {
                subgraph: {
                    node: [dict(neighbor) for neighbor in neighbors]
                    for node, neighbors in adjacency.items()
                }
                for subgraph, adjacency in self.local_adjacencies.items()
            },
            "inter_subgraph_edges": [dict(edge) for edge in self.inter_subgraph_edges],
        }

    def save_json(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def load_json(self, path: str | Path) -> None:
        input_path = Path(path)
        payload = json.loads(input_path.read_text(encoding="utf-8"))
        self.load_from_dict(payload)

    def map_entities_to_communities(self, *node_names: Any) -> Dict[Any, str]:
        if len(node_names) == 1 and isinstance(node_names[0], Iterable) and not isinstance(
            node_names[0], (str, bytes)
        ):
            values = list(node_names[0])
        else:
            values = list(node_names)

        result: Dict[Any, str] = {}
        for node_name in values:
            key = str(node_name)
            if key not in self.node_to_subgraph:
                raise KeyError(f"Node '{key}' is not present in graph memory.")
            result[node_name] = self.node_to_subgraph[key]
        return result

    def graph_extract(self, subgraph_id: Any, hop: int = 1) -> Dict[str, Any]:
        if hop < 0:
            raise ValueError("hop must be >= 0")

        key = str(subgraph_id)
        if key not in self.meta_graph:
            known_subgraphs = set(self.node_to_subgraph.values())
            if key in known_subgraphs:
                return {
                    key: [],
                    "_info": (
                        f"Subgraph {key} has no inter-subgraph connections. "
                        "Solve locally inside this subgraph."
                    ),
                }
            raise KeyError(f"Subgraph '{key}' does not exist in graph memory.")

        visited: Dict[str, int] = {key: 0}
        queue: deque[str] = deque([key])
        sliced: Dict[str, List[str]] = {}

        while queue:
            current = queue.popleft()
            depth = visited[current]
            neighbors = self.meta_graph.get(current, [])
            sliced[current] = []
            for neighbor in neighbors:
                neighbor_key = str(neighbor)
                if neighbor_key not in visited and depth < hop:
                    visited[neighbor_key] = depth + 1
                    queue.append(neighbor_key)
                if neighbor_key in visited and visited[neighbor_key] <= hop:
                    sliced[current].append(neighbor_key)

        return {
            subgraph: sort_subgraph_ids(neighbors)
            for subgraph, neighbors in sliced.items()
        }

    def _build_local_graph(self, subgraph_id: str) -> nx.Graph:
        if subgraph_id in self._local_graph_cache:
            return self._local_graph_cache[subgraph_id]

        adjacency = self.local_adjacencies.get(subgraph_id, {})
        local_graph = nx.Graph()
        for node in adjacency.keys():
            local_graph.add_node(node)
        for node, neighbors in adjacency.items():
            for edge in neighbors:
                target = str(edge.get("target", ""))
                if not target:
                    continue
                local_graph.add_edge(node, target)

        self._local_graph_cache[subgraph_id] = local_graph
        return local_graph

    def local_shortest_path(
        self, subgraph_id: Any, source: str, target: str
    ) -> Optional[List[str]]:
        key = str(subgraph_id)
        source_name = str(source)
        target_name = str(target)
        if source_name == target_name:
            if source_name in self.local_adjacencies.get(key, {}):
                return [source_name]
            return None

        local_graph = self._build_local_graph(key)
        if source_name not in local_graph or target_name not in local_graph:
            return None

        try:
            return list(nx.shortest_path(local_graph, source_name, target_name))
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def subgraph_neighbors(self, subgraph_id: Any) -> List[str]:
        key = str(subgraph_id)
        return sort_subgraph_ids(self.meta_graph.get(key, []))

    def candidate_portal_links(
        self, current_subgraph: Any, next_subgraph: Any
    ) -> List[PortalLink]:
        current = str(current_subgraph)
        nxt = str(next_subgraph)
        links: List[PortalLink] = []
        seen: set[tuple[str, str, str, str, str]] = set()

        for edge in self.inter_subgraph_edges:
            source_subgraph = str(edge.get("source_subgraph"))
            target_subgraph = str(edge.get("target_subgraph"))
            relation = str(edge.get("relation", "connected_to"))

            if source_subgraph == current and target_subgraph == nxt:
                portal_node = str(edge.get("source"))
                entry_node = str(edge.get("target"))
            elif source_subgraph == nxt and target_subgraph == current:
                portal_node = str(edge.get("target"))
                entry_node = str(edge.get("source"))
            else:
                continue

            record_key = (current, nxt, portal_node, entry_node, relation)
            if record_key in seen:
                continue
            seen.add(record_key)
            links.append(
                PortalLink(
                    current_subgraph=current,
                    next_subgraph=nxt,
                    portal_node=portal_node,
                    entry_node=entry_node,
                    relation=relation,
                )
            )

        links.sort(key=lambda item: (item.portal_node, item.entry_node, item.relation))
        return links

    def get_subgraph_relationships(
        self,
        current_subgraph: Any,
        target_subgraph: Any,
        allowed_portals: Optional[Iterable[str]] = None,
    ) -> List[str]:
        allowed = {str(node) for node in allowed_portals} if allowed_portals is not None else None
        links = self.candidate_portal_links(current_subgraph, target_subgraph)
        relationships: List[str] = []
        current = str(current_subgraph)
        target = str(target_subgraph)
        for link in links:
            if allowed is not None and link.portal_node not in allowed:
                continue
            relationships.append(
                f"Subgraph {current} --{link.relation}--> {link.entry_node} (Subgraph {target})"
            )
        return relationships
