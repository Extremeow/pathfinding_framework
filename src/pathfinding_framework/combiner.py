from __future__ import annotations

from typing import Any, Dict, Iterable, List


def combine_worker_paths(
    worker_records: Iterable[Dict[str, Any]],
    source: str,
    target: str,
) -> Dict[str, Any]:
    """
    Combine solver segments into one end-to-end path.

    Concatenation rule:
    - if previous tail equals next head, remove duplicated node.
    """

    records = list(worker_records)
    if not records:
        return {
            "final_path": [],
            "status": "failed",
            "confidence": 0.0,
            "explanation": "No worker records were available for combination.",
        }

    combined: List[str] = []
    for record in records:
        result = record.get("result", {})
        paths_found = result.get("paths_found") or []
        if not paths_found or not paths_found[0]:
            continue
        segment = [str(node) for node in paths_found[0]]
        if not combined:
            combined.extend(segment)
        elif combined[-1] == segment[0]:
            combined.extend(segment[1:])
        else:
            combined.extend(segment)

    if not combined:
        return {
            "final_path": [],
            "status": "failed",
            "confidence": 0.0,
            "explanation": "All worker records were empty.",
        }

    status = "success" if combined[0] == source and combined[-1] == target else "partial"
    confidence = 1.0 if status == "success" else 0.5

    return {
        "final_path": combined,
        "status": status,
        "confidence": confidence,
        "explanation": f"Combined {len(records)} worker segment(s) deterministically.",
    }
