# Pathfinding Framework

Clean, deterministic reconstruction of the original multi-agent pathfinding project.

This version keeps the core ideas from `D:\reconstruction\Framework\pathfinding_framework.py`:

- graph partitioning into subgraphs
- metagraph memory (portal nodes, local adjacency, inter-subgraph edges)
- subgraph-level planning
- per-segment solver dispatch records
- final path combination
- path validation and CLI execution

It intentionally removes the old LLM orchestration complexity and replaces it with explicit, testable Python logic.

## Key Features

- deterministic hierarchical pathfinding pipeline
- optional Leiden partitioning (`igraph` + `leidenalg`) with greedy modularity fallback
- JSON metagraph save/load
- compatibility with dataset pickle files used by the legacy framework
- end-to-end CLI
- pytest coverage for core flows

## Project Layout

```text
pathfinding_framework/
  src/pathfinding_framework/
    framework.py       # Main API: initialize, load/save metagraph, solve
    partitioning.py    # Graph partition + metagraph construction
    memory.py          # Runtime graph memory operations
    solver.py          # Deterministic local solver dispatch
    combiner.py        # Path stitching
    planner.py         # Subgraph route planning
    validation.py      # Final path checks
    datasets.py        # Legacy dataset loading helpers
    cli.py             # Command line entrypoint
  tests/
```

## Install

```bash
pip install -e .
```

Optional Leiden dependencies:

```bash
pip install -e .[leiden]
```

## Python Usage

```python
import pickle
from pathfinding_framework import PathfindingFramework

with open("dataset/Distance_100.pkl", "rb") as f:
    problems = pickle.load(f)

problem = problems[0]
graph = problem["graph"]
source = graph.nodes[problem["source"]].get("name", str(problem["source"]))
target = graph.nodes[problem["target"]].get("name", str(problem["target"]))

framework = PathfindingFramework(verbose=True)
framework.initialize_graph(graph, resolution=3.5, method="auto")

result = framework.solve(
    source=source,
    target=target,
    graph=graph,
    expected_distance=problem.get("exact_answer"),
)

print(result["status"], result["final_path"])
```

## CLI

```bash
pfw-run --dataset D:\reconstruction\Framework\dataset\Distance_100.pkl --problem 0
```

Main options:

- `--partition-method auto|leiden|greedy`
- `--resolution 3.5`
- `--save-metagraph <path>`
- `--load-metagraph <path>`
- `--no-preprocess`
- `--quiet`

## Design Notes

- Node names are canonical runtime IDs. Duplicate node names are rejected at initialization.
- The solver works per subgraph route and selects portal transitions deterministically.
- If hierarchical solving fails but full graph is available, the framework falls back to global shortest path and reports `success_fallback`.

## Testing

```bash
pytest
```

## Scope of Reconstruction

This repository is a structural and code-level reconstruction of the framework itself (not a file-for-file copy of the legacy codebase).  
The architecture was redesigned for readability, maintainability, and deterministic behavior.
