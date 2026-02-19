from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class PortalLink:
    """Represents one directed inter-subgraph transition."""

    current_subgraph: str
    next_subgraph: str
    portal_node: str
    entry_node: str
    relation: str


@dataclass
class WorkerDispatchRecord:
    """Normalized record for one local solver dispatch."""

    dispatch_id: int
    subgraph_id: str
    source: str
    target: str
    priority_subgraph: Optional[str]
    result: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dispatch_id": self.dispatch_id,
            "subgraph_id": self.subgraph_id,
            "source": self.source,
            "target": self.target,
            "priority_subgraph": self.priority_subgraph,
            "result": self.result,
        }
