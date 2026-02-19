# Pathfinding Framework

Clean LLM-first reconstruction of the original multi-agent pathfinding system.

This version keeps the core architecture from `D:\reconstruction\Framework\pathfinding_framework.py`:

- Master orchestration with tool-calling
- Solver agent for local subgraph path search
- Combiner agent for final path stitching
- Leiden/greedy graph partitioning + metagraph memory
- Path validation against the real graph

The focus is reducing overlap and dead code while preserving the LLM workflow.

## Architecture

- `Master` (LLM): decides which tools to call and in what order.
- `dispatch_solver` (LLM): solves one subgraph step at a time.
- `get_subgraph_relationships`: exposes portal transitions from last dispatched subgraph.
- `dispatch_combiner` (LLM): merges worker segments into one path.

## Project Layout

```text
pathfinding_framework/
  src/pathfinding_framework/
    framework.py       # Master loop and tool execution
    solver.py          # LLM solver agent
    combiner.py        # LLM combiner agent
    llm_client.py      # OpenAI client wrapper + usage tracking
    memory.py          # Metagraph memory store
    partitioning.py    # Leiden/greedy partitioning
    validation.py      # Path validation
    datasets.py        # Dataset loading helpers
    cli.py             # CLI entrypoint
```

## Install

```bash
pip install -e .
```

Optional Leiden dependencies:

```bash
pip install -e .[leiden]
```

## API Key

Set your key in environment:

```bash
set OPENAI_API_KEY=your_key_here
```

or pass CLI flag:

```bash
pfw-run --api-key your_key_here ...
```

## CLI Usage

```bash
pfw-run --dataset D:\reconstruction\Framework\dataset\Distance_100.pkl --problem 0
```

Useful options:

- `--model gpt-4.1-2025-04-14`
- `--max-turns 30`
- `--partition-method auto|leiden|greedy`
- `--resolution 3.5`
- `--save-metagraph <path>`
- `--load-metagraph <path>`
- `--no-preprocess`
- `--quiet`

## Python Usage

```python
import pickle
from pathfinding_framework import PathfindingFramework

with open(r"D:\reconstruction\Framework\dataset\Distance_100.pkl", "rb") as f:
    problems = pickle.load(f)

problem = problems[0]
graph = problem["graph"]
source = graph.nodes[problem["source"]].get("name", str(problem["source"]))
target = graph.nodes[problem["target"]].get("name", str(problem["target"]))

framework = PathfindingFramework(
    api_key=None,  # uses OPENAI_API_KEY if set
    model="gpt-4.1-2025-04-14",
    max_turns=30,
    verbose=True,
)
framework.initialize_graph(graph, method="auto", resolution=3.5)

result = framework.solve(
    source=source,
    target=target,
    graph=graph,
    expected_distance=problem.get("exact_answer"),
)
print(result["status"], result["final_path"])
```
