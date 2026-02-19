from __future__ import annotations

from typing import Optional, Tuple

from .memory import GraphMemory
from .models import PortalLink, WorkerDispatchRecord
from .utils import sort_subgraph_ids


def _validate_source(memory: GraphMemory, subgraph_id: str, source: str) -> Optional[str]:
    if source not in memory.node_to_subgraph:
        return f"Source node '{source}' is not in graph memory."
    expected_subgraph = memory.node_to_subgraph[source]
    if expected_subgraph != subgraph_id:
        return (
            f"Source node '{source}' belongs to subgraph '{expected_subgraph}', "
            f"not '{subgraph_id}'."
        )
    return None


def _choose_transition(
    memory: GraphMemory, subgraph_id: str, source: str, next_subgraph: str
) -> Tuple[Optional[PortalLink], Optional[list[str]]]:
    best_link: Optional[PortalLink] = None
    best_path: Optional[list[str]] = None
    best_score: Optional[tuple[int, str, str, str]] = None

    for link in memory.candidate_portal_links(subgraph_id, next_subgraph):
        if source == link.portal_node:
            path = [source]
        else:
            path = memory.local_shortest_path(subgraph_id, source, link.portal_node)
        if not path:
            continue

        score = (len(path) - 1, link.portal_node, link.entry_node, link.relation)
        if best_score is None or score < best_score:
            best_score = score
            best_link = link
            best_path = path

    return best_link, best_path


def dispatch_solver(
    memory: GraphMemory,
    dispatch_id: int,
    subgraph_id: str,
    source: str,
    target: str,
    priority_subgraph: str | None,
) -> WorkerDispatchRecord:
    """
    Deterministic local solver dispatch.

    Rules:
    - If `priority_subgraph == subgraph_id`, solve directly to final target.
    - Otherwise choose the shortest local path to a portal connected to priority_subgraph.
    """

    subgraph_key = str(subgraph_id)
    priority_key = str(priority_subgraph) if priority_subgraph is not None else None
    source_name = str(source)
    target_name = str(target)

    validation_error = _validate_source(memory, subgraph_key, source_name)
    if validation_error:
        return WorkerDispatchRecord(
            dispatch_id=dispatch_id,
            subgraph_id=subgraph_key,
            source=source_name,
            target=target_name,
            priority_subgraph=priority_key,
            result={
                "status": "failed",
                "paths_found": [],
                "portal_nodes": [],
                "subgraph_connections": {},
                "connected_subgraphs": sort_subgraph_ids(memory.subgraph_neighbors(subgraph_key)),
                "error": validation_error,
            },
        )

    connected_subgraphs = memory.subgraph_neighbors(subgraph_key)

    if priority_key is None or priority_key == subgraph_key:
        if source_name == target_name:
            path = [source_name]
        else:
            path = memory.local_shortest_path(subgraph_key, source_name, target_name)

        if not path:
            result = {
                "status": "failed",
                "paths_found": [],
                "portal_nodes": [],
                "subgraph_connections": {},
                "connected_subgraphs": connected_subgraphs,
                "error": f"No local path from '{source_name}' to '{target_name}' in subgraph '{subgraph_key}'.",
                "assignment": {
                    "target_subgraph": priority_key or subgraph_key,
                    "portal_node": None,
                    "status": "failed",
                },
            }
        else:
            result = {
                "status": "reached",
                "paths_found": [path],
                "portal_nodes": [],
                "subgraph_connections": {},
                "connected_subgraphs": connected_subgraphs,
                "assignment": {
                    "target_subgraph": priority_key or subgraph_key,
                    "portal_node": None,
                    "status": "complete",
                },
            }
        return WorkerDispatchRecord(
            dispatch_id=dispatch_id,
            subgraph_id=subgraph_key,
            source=source_name,
            target=target_name,
            priority_subgraph=priority_key,
            result=result,
        )

    transition, path_to_portal = _choose_transition(
        memory=memory,
        subgraph_id=subgraph_key,
        source=source_name,
        next_subgraph=priority_key,
    )

    if transition is None or not path_to_portal:
        return WorkerDispatchRecord(
            dispatch_id=dispatch_id,
            subgraph_id=subgraph_key,
            source=source_name,
            target=target_name,
            priority_subgraph=priority_key,
            result={
                "status": "failed",
                "paths_found": [],
                "portal_nodes": [],
                "subgraph_connections": {},
                "connected_subgraphs": connected_subgraphs,
                "error": (
                    f"No reachable portal from source '{source_name}' in subgraph '{subgraph_key}' "
                    f"to target subgraph '{priority_key}'."
                ),
                "assignment": {
                    "target_subgraph": priority_key,
                    "portal_node": None,
                    "status": "failed",
                },
            },
        )

    result = {
        "status": "complete",
        "paths_found": [path_to_portal],
        "portal_nodes": [transition.portal_node],
        "subgraph_connections": {
            priority_key: [
                {
                    "relation": transition.relation,
                    "linked_node": transition.entry_node,
                    "portal": transition.portal_node,
                }
            ]
        },
        "connected_subgraphs": connected_subgraphs,
        "selected_transition": {
            "from_subgraph": subgraph_key,
            "to_subgraph": priority_key,
            "portal_node": transition.portal_node,
            "linked_node": transition.entry_node,
            "relation": transition.relation,
        },
        "assignment": {
            "target_subgraph": priority_key,
            "portal_node": transition.portal_node,
            "status": "complete",
        },
    }

    return WorkerDispatchRecord(
        dispatch_id=dispatch_id,
        subgraph_id=subgraph_key,
        source=source_name,
        target=target_name,
        priority_subgraph=priority_key,
        result=result,
    )
