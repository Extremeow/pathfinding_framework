from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PortalLink:
    """Represents one directed inter-subgraph transition."""

    current_subgraph: str
    next_subgraph: str
    portal_node: str
    entry_node: str
    relation: str
