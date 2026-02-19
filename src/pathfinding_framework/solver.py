from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterable, List, Tuple

from .llm_client import OpenAIChatClient
from .memory import GraphMemory
from .prompt_loader import load_prompt_text
from .utils import sort_subgraph_ids


DEFAULT_SOLVER_SYSTEM_PROMPT = """You are a local pathfinding solver.

You get one subgraph adjacency list and must output JSON only.
Do not include markdown fences or extra text.

Output schema:
{
  "status": "reached" | "complete" | "failed",
  "paths_found": [["node1", "node2", "..."]],
  "portal_nodes": ["portalA", "portalB"],
  "confidence": 0.0-1.0,
  "reason": "short rationale",
  "cot_steps": ["step 1 reasoning", "step 2 reasoning"]
}

Rules:
- If target is inside current subgraph, find a path to target and set status="reached".
- Otherwise find a path to a portal that exits to the requested next subgraph and set status="complete".
- If no valid path exists, set status="failed" and paths_found=[].
- Think step-by-step before deciding the path.
- Keep cot_steps concise and factual (3-8 items).
- If retry_feedback is provided in input, fix those specific issues.
"""
SOLVER_SYSTEM_PROMPT = load_prompt_text("solver_prompt.txt", DEFAULT_SOLVER_SYSTEM_PROMPT)


def _extract_json(text: str) -> Dict[str, Any] | None:
    text = (text or "").strip()
    if not text:
        return None

    try:
        value = json.loads(text)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        value = json.loads(match.group(0))
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        return None
    return None


def _normalize_paths(raw_paths: Any) -> List[List[str]]:
    if not isinstance(raw_paths, list):
        return []
    parsed: List[List[str]] = []
    for entry in raw_paths:
        if not isinstance(entry, list):
            continue
        path = [str(node) for node in entry if str(node).strip()]
        if path:
            parsed.append(path)
    return parsed


def _validate_local_path(memory: GraphMemory, subgraph_id: str, path: Iterable[str]) -> bool:
    adjacency = memory.local_adjacencies.get(subgraph_id, {})
    nodes = list(path)
    if not nodes:
        return False
    if any(node not in adjacency for node in nodes):
        return False
    for idx in range(len(nodes) - 1):
        source = nodes[idx]
        target = nodes[idx + 1]
        neighbors = adjacency.get(source, [])
        if not any(str(edge.get("target")) == target for edge in neighbors):
            return False
    return True


def _portal_targets(memory: GraphMemory, subgraph_id: str) -> Dict[str, set[str]]:
    subgraph_key = str(subgraph_id)
    mapping: Dict[str, set[str]] = {}
    for edge in memory.inter_subgraph_edges:
        source = str(edge.get("source"))
        target = str(edge.get("target"))
        source_subgraph = str(edge.get("source_subgraph"))
        target_subgraph = str(edge.get("target_subgraph"))
        if source_subgraph == subgraph_key:
            mapping.setdefault(source, set()).add(target_subgraph)
        elif target_subgraph == subgraph_key:
            mapping.setdefault(target, set()).add(source_subgraph)
    return mapping


def _build_subgraph_connections(
    memory: GraphMemory,
    subgraph_id: str,
    portal_nodes: Iterable[str],
) -> Dict[str, List[Dict[str, str]]]:
    subgraph_key = str(subgraph_id)
    selected = {str(node) for node in portal_nodes}
    connections: Dict[str, List[Dict[str, str]]] = {}
    seen: set[Tuple[str, str, str, str]] = set()

    for edge in memory.inter_subgraph_edges:
        source = str(edge.get("source"))
        target = str(edge.get("target"))
        source_subgraph = str(edge.get("source_subgraph"))
        target_subgraph = str(edge.get("target_subgraph"))
        relation = str(edge.get("relation", "connected_to"))

        if source_subgraph == subgraph_key and source in selected:
            next_subgraph = target_subgraph
            portal = source
            linked_node = target
        elif target_subgraph == subgraph_key and target in selected:
            next_subgraph = source_subgraph
            portal = target
            linked_node = source
        else:
            continue

        signature = (next_subgraph, relation, linked_node, portal)
        if signature in seen:
            continue
        seen.add(signature)
        connections.setdefault(next_subgraph, []).append(
            {
                "relation": relation,
                "linked_node": linked_node,
                "portal": portal,
            }
        )

    ordered: Dict[str, List[Dict[str, str]]] = {}
    for subgraph in sort_subgraph_ids(connections.keys()):
        ordered[subgraph] = connections[subgraph]
    return ordered


def _serialize_subgraph_adjacency(
    memory: GraphMemory,
    subgraph_id: str,
    priority_subgraph: str | None,
) -> str:
    subgraph_key = str(subgraph_id)
    adjacency = memory.local_adjacencies.get(subgraph_key, {})
    portal_map = _portal_targets(memory, subgraph_key)
    priority = str(priority_subgraph) if priority_subgraph is not None else None

    lines: List[str] = []
    for node in sorted(adjacency.keys()):
        node_label = node
        if node in portal_map:
            targets = sort_subgraph_ids(portal_map[node])
            if priority and priority in targets:
                node_label = f"{node} [PORTAL:{priority}]"
            elif not priority:
                node_label = f"{node} [PORTAL:{','.join(targets)}]"

        neighbors = []
        for edge in adjacency[node]:
            target = str(edge.get("target"))
            relation = str(edge.get("relation", "connected_to"))
            target_label = target
            if target in portal_map:
                targets = sort_subgraph_ids(portal_map[target])
                if priority and priority in targets:
                    target_label = f"{target} [PORTAL:{priority}]"
                elif not priority:
                    target_label = f"{target} [PORTAL:{','.join(targets)}]"
            neighbors.append(f"--{relation}--> {target_label}")
        lines.append(f"{node_label}: {', '.join(neighbors)}")
    return "\n".join(lines)


def run_solver_agent(
    *,
    client: OpenAIChatClient,
    memory: GraphMemory,
    subgraph_id: str,
    source: str,
    target: str,
    priority_subgraph: str | None,
    retry_feedback: str | None = None,
) -> Dict[str, Any]:
    subgraph_key = str(subgraph_id)
    source_name = str(source)
    target_name = str(target)
    priority = str(priority_subgraph) if priority_subgraph is not None else None

    local_adj = memory.local_adjacencies.get(subgraph_key, {})
    if source_name not in local_adj:
        return {
            "status": "failed",
            "paths_found": [],
            "portal_nodes": [],
            "subgraph_connections": {},
            "error": f"Source '{source_name}' is not in subgraph '{subgraph_key}'.",
        }

    target_in_subgraph = target_name in local_adj
    if priority == subgraph_key:
        target_in_subgraph = True

    user_payload = {
        "current_subgraph": subgraph_key,
        "source": source_name,
        "target": target_name,
        "priority_subgraph": priority,
        "target_in_current_subgraph": target_in_subgraph,
        "adjacency": _serialize_subgraph_adjacency(memory, subgraph_key, priority),
        "retry_feedback": retry_feedback or "",
    }
    messages = [
        {"role": "system", "content": SOLVER_SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False, indent=2)},
    ]

    response = client.chat_completion(messages=messages, temperature=0.0)
    raw_output = response.choices[0].message.content or ""
    parsed = _extract_json(raw_output)
    if parsed is None:
        return {
            "status": "failed",
            "paths_found": [],
            "portal_nodes": [],
            "subgraph_connections": {},
            "error": "Solver output was not valid JSON.",
            "raw_output": raw_output,
        }

    status = str(parsed.get("status", "failed")).lower()
    if status not in {"reached", "complete", "failed"}:
        status = "failed"

    raw_paths = _normalize_paths(parsed.get("paths_found"))
    paths_found = [path for path in raw_paths if _validate_local_path(memory, subgraph_key, path)]
    if not paths_found and status in {"reached", "complete"}:
        status = "failed"

    portal_candidates = []
    portal_nodes_field = parsed.get("portal_nodes")
    if isinstance(portal_nodes_field, list):
        portal_candidates.extend(str(node) for node in portal_nodes_field)
    elif parsed.get("portal_node"):
        portal_candidates.append(str(parsed.get("portal_node")))

    portals_known = set(memory.portal_nodes.get(subgraph_key, []))
    portal_nodes = [node for node in portal_candidates if node in portals_known]

    if status == "complete" and not portal_nodes and paths_found:
        terminal = paths_found[0][-1]
        if terminal in portals_known:
            portal_nodes = [terminal]

    if priority and status == "complete":
        portal_targets = _portal_targets(memory, subgraph_key)
        portal_nodes = [
            portal for portal in portal_nodes if priority in portal_targets.get(portal, set())
        ]
        if not portal_nodes:
            status = "failed"

    subgraph_connections = _build_subgraph_connections(memory, subgraph_key, portal_nodes)
    connected_subgraphs = sort_subgraph_ids(subgraph_connections.keys())

    result: Dict[str, Any] = {
        "status": status,
        "paths_found": paths_found,
        "portal_nodes": portal_nodes,
        "subgraph_connections": subgraph_connections,
        "connected_subgraphs": connected_subgraphs,
        "confidence": float(parsed.get("confidence", 0.0) or 0.0),
        "reason": str(parsed.get("reason", "")).strip(),
        "raw_output": raw_output,
    }
    cot_steps = parsed.get("cot_steps")
    if isinstance(cot_steps, list):
        result["cot_steps"] = [str(x) for x in cot_steps if str(x).strip()][:8]

    if priority:
        assignment_status = "complete" if status in {"reached", "complete"} else "failed"
        result["assignment"] = {
            "target_subgraph": priority,
            "portal_node": portal_nodes[0] if portal_nodes else None,
            "status": assignment_status,
        }

    return result
