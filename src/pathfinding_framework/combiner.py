from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterable, List

from .llm_client import OpenAIChatClient
from .prompt_loader import load_prompt_text


DEFAULT_COMBINER_SYSTEM_PROMPT = """You are a path combiner agent.

Input is an ordered list of worker dispatch outputs from subgraph solvers.
Combine them into one final path from source to target.

Return JSON only:
{
  "final_path": ["..."],
  "status": "success" | "partial" | "failed",
  "confidence": 0.0-1.0,
  "explanation": "brief reason"
}

Rules:
- Prefer the first valid path in each worker record.
- If consecutive segments share a boundary node, avoid duplicating it.
- Do not invent nodes not present in worker outputs.
"""
COMBINER_SYSTEM_PROMPT = load_prompt_text("combiner_prompt.txt", DEFAULT_COMBINER_SYSTEM_PROMPT)


def _extract_json(text: str) -> Dict[str, Any] | None:
    content = (text or "").strip()
    if not content:
        return None
    try:
        data = json.loads(content)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", content, flags=re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def run_combiner_agent(
    *,
    client: OpenAIChatClient,
    worker_records: Iterable[Dict[str, Any]],
    source: str,
    target: str,
) -> Dict[str, Any]:
    records = list(worker_records)
    if not records:
        return {
            "final_path": [],
            "status": "failed",
            "confidence": 0.0,
            "explanation": "No worker records available.",
        }

    simplified: List[Dict[str, Any]] = []
    for record in records:
        payload = record.get("result", {})
        simplified.append(
            {
                "dispatch_id": record.get("dispatch_id"),
                "subgraph_id": record.get("subgraph_id"),
                "source": record.get("source"),
                "priority_subgraph": record.get("priority_subgraph"),
                "status": payload.get("status"),
                "paths_found": payload.get("paths_found", []),
                "portal_nodes": payload.get("portal_nodes", []),
                "subgraph_connections": payload.get("subgraph_connections", {}),
            }
        )

    messages = [
        {"role": "system", "content": COMBINER_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": json.dumps(
                {
                    "source": str(source),
                    "target": str(target),
                    "worker_records": simplified,
                },
                ensure_ascii=False,
                indent=2,
            ),
        },
    ]

    response = client.chat_completion(messages=messages, temperature=0.0)
    raw_output = response.choices[0].message.content or ""
    parsed = _extract_json(raw_output)
    if parsed is None:
        return {
            "final_path": [],
            "status": "failed",
            "confidence": 0.0,
            "explanation": "Combiner output was not valid JSON.",
            "raw_output": raw_output,
        }

    final_path_field = parsed.get("final_path", [])
    if isinstance(final_path_field, list):
        final_path = [str(node) for node in final_path_field if str(node).strip()]
    else:
        final_path = []

    status = str(parsed.get("status", "failed")).lower()
    if status not in {"success", "partial", "failed"}:
        status = "failed"

    return {
        "final_path": final_path,
        "status": status,
        "confidence": float(parsed.get("confidence", 0.0) or 0.0),
        "explanation": str(parsed.get("explanation", "")).strip(),
        "raw_output": raw_output,
    }
