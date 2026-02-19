from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from .datasets import load_problem_set, resolve_problem
from .framework import PathfindingFramework
from .llm_client import DEFAULT_MODEL
from .partitioning import DEFAULT_RESOLUTION


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LLM-based hierarchical pathfinding framework")
    parser.add_argument("--dataset", type=str, required=True, help="Path to dataset pickle file")
    parser.add_argument("--problem", type=int, default=0, help="Problem index to solve")
    parser.add_argument("--api-key", type=str, help="OpenAI API key (optional if OPENAI_API_KEY is set)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="OpenAI model for master/solver/combiner")
    parser.add_argument("--max-turns", type=int, default=30, help="Maximum master turns")
    parser.add_argument(
        "--partition-method",
        type=str,
        default="auto",
        choices=["auto", "leiden", "greedy"],
        help="Partition algorithm selection",
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=DEFAULT_RESOLUTION,
        help="Leiden resolution parameter (used when Leiden is active)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for partitioning")
    parser.add_argument("--save-metagraph", type=str, help="Path to save generated metagraph JSON")
    parser.add_argument("--load-metagraph", type=str, help="Path to load metagraph JSON")
    parser.add_argument("--no-preprocess", action="store_true", help="Disable largest-component preprocessing")
    parser.add_argument("--quiet", action="store_true", help="Disable verbose logs")
    return parser


def _print_result(result: dict, expected_distance: Optional[int]) -> None:
    print("\n" + "=" * 80)
    print("RESULT")
    print("=" * 80)
    print(f"Status: {result['status']}")
    print(f"Source: {result['source']} (Subgraph {result['source_subgraph']})")
    print(f"Target: {result['target']} (Subgraph {result['target_subgraph']})")
    print(f"Dispatches: {result['worker_count']}")
    print(f"Strategy: {result['strategy']}")

    final_path = result.get("final_path", [])
    if final_path:
        print(f"Final path ({len(final_path)} nodes):")
        print("  " + " -> ".join(final_path))
    else:
        print("Final path: <none>")

    validation = result.get("validation")
    if validation:
        print("\nValidation:")
        print(f"  is_valid: {validation.get('is_valid')}")
        print(f"  path_length: {validation.get('path_length')}")
        if expected_distance is not None:
            print(f"  expected_distance: {expected_distance}")
            print(f"  distance_match: {validation.get('distance_match')}")
        if validation.get("errors"):
            print(f"  errors: {validation['errors']}")

    token_usage = result.get("token_usage") or {}
    if token_usage:
        print("\nToken usage:")
        print(f"  calls: {token_usage.get('calls', 0)}")
        print(f"  input_tokens: {token_usage.get('input_tokens', 0)}")
        print(f"  output_tokens: {token_usage.get('output_tokens', 0)}")
        print(f"  total_tokens: {token_usage.get('total_tokens', 0)}")
        print(f"  est_cost_usd: {token_usage.get('cost', 0.0):.6f}")

    print(f"\nSolve time: {result.get('solve_time_seconds', 0.0):.3f}s")
    print("=" * 80)


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    problems = load_problem_set(Path(args.dataset))
    if args.problem < 0 or args.problem >= len(problems):
        parser.error(f"--problem must be in [0, {len(problems) - 1}]")

    graph, source, target, expected_distance = resolve_problem(problems[args.problem])

    framework = PathfindingFramework(
        api_key=args.api_key,
        model=args.model,
        max_turns=args.max_turns,
        verbose=not args.quiet,
    )

    if args.load_metagraph:
        framework.load_graph_memory(args.load_metagraph)
    else:
        framework.initialize_graph(
            graph=graph,
            resolution=args.resolution,
            seed=args.seed,
            method=args.partition_method,
            preprocess=not args.no_preprocess,
            save_path=args.save_metagraph,
        )

    try:
        result = framework.solve(
            source=source,
            target=target,
            graph=graph,
            expected_distance=expected_distance,
        )
    except RuntimeError as exc:
        print(f"Error: {exc}")
        return 2
    _print_result(result, expected_distance)

    return 0 if result["status"] in {"success", "partial"} else 1
