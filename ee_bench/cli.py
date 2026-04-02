"""CLI entry point for ee-bench."""

from __future__ import annotations

import argparse
import json
import os
import sys

from .config import ExperimentConfig, SweepConfig
from .environments import ALL_ENVIRONMENTS
from .runner import ENV_MAP, run_single, run_sweep, save_results


def _env_list() -> str:
    lines = []
    for env in ALL_ENVIRONMENTS:
        stat = "stationary" if env.is_stationary else "non-stationary"
        lines.append(f"  {env.name:<25} {env.description} [{stat}]")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="ee-bench: Explore/Exploit Benchmark for LLMs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Available environments:\n{_env_list()}",
    )
    sub = parser.add_subparsers(dest="command")

    # --- run: single experiment ---
    run_p = sub.add_parser("run", help="Run a single experiment")
    run_p.add_argument("--model", required=True, help="OpenRouter model id")
    run_p.add_argument("--env", required=True, help="Environment name")
    run_p.add_argument("--temperature", type=float, default=0.7)
    run_p.add_argument("--horizons", type=int, nargs="+", default=[20, 100, 500])
    run_p.add_argument("--repetitions", type=int, default=5)
    run_p.add_argument("--seed", type=int, default=42)
    run_p.add_argument("--output", default="results")
    run_p.add_argument("--api-key", default=None)
    run_p.add_argument("--base-url", default="https://openrouter.ai/api/v1")

    # --- sweep: full parameter sweep ---
    sweep_p = sub.add_parser("sweep", help="Run a parameter sweep from a config file")
    sweep_p.add_argument("config_file", help="JSON config file for the sweep")
    sweep_p.add_argument("--api-key", default=None)

    # --- list: show environments ---
    sub.add_parser("list", help="List available environments")

    args = parser.parse_args()

    if args.command == "list":
        print("Available environments:\n")
        print(_env_list())
        return

    if args.command == "run":
        api_key = args.api_key or os.environ.get("OPENROUTER_API_KEY", "")
        if not api_key:
            print("Error: set OPENROUTER_API_KEY or pass --api-key", file=sys.stderr)
            sys.exit(1)

        config = ExperimentConfig(
            model=args.model,
            environment=args.env,
            temperature=args.temperature,
            horizons=args.horizons,
            repetitions=args.repetitions,
            seed=args.seed,
            api_key=api_key,
            base_url=args.base_url,
            results_dir=args.output,
        )
        results = run_single(config)
        save_results(results, config.results_dir)

    elif args.command == "sweep":
        with open(args.config_file) as f:
            raw = json.load(f)

        api_key = args.api_key or os.environ.get("OPENROUTER_API_KEY", "")
        if not api_key:
            print("Error: set OPENROUTER_API_KEY or pass --api-key", file=sys.stderr)
            sys.exit(1)

        config = SweepConfig(
            models=raw["models"],
            temperatures=raw.get("temperatures", [0.0, 0.3, 0.7, 1.0]),
            environments=raw.get("environments", []),
            horizons=raw.get("horizons", [20, 100, 500]),
            repetitions=raw.get("repetitions", 5),
            seed=raw.get("seed", 42),
            api_key=api_key,
            base_url=raw.get("base_url", "https://openrouter.ai/api/v1"),
            results_dir=raw.get("results_dir", "results"),
        )
        results = run_sweep(config)
        save_results(results, config.results_dir, name="sweep")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
