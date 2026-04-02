"""Analysis functions for ee-bench results."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def summary_table(results: list[dict], out=None):
    """Print a summary table of all runs."""
    p = lambda *a, **k: print(*a, **k, file=out)
    p(f"\n{'Model':<40} {'Env':<25} {'Temp':>5} {'Horizon':>8} {'AvgReward':>10} {'AvgRegret':>10} {'ExplRatio':>10} {'UniqueAct':>10}")
    p("=" * 130)

    for run in results:
        model = run["model"]
        env = run["environment"]
        temp = run["temperature"]

        by_horizon: dict[int, list] = defaultdict(list)
        for ep in run["episodes"]:
            by_horizon[ep["horizon"]].append(ep["metrics"])

        for horizon, metrics_list in sorted(by_horizon.items()):
            avg_reward = np.mean([m["mean_reward"] for m in metrics_list])
            avg_regret = np.mean([m["total_regret"] for m in metrics_list])
            avg_expl = np.mean([m["final_exploration_ratio"] for m in metrics_list])
            avg_unique = np.mean([m["unique_actions_tried"] for m in metrics_list])
            p(f"{model:<40} {env:<25} {temp:>5.1f} {horizon:>8} {avg_reward:>10.4f} {avg_regret:>10.2f} {avg_expl:>10.3f} {avg_unique:>10.1f}")


def compare_models(results: list[dict], out=None):
    """Compare models across environments."""
    p = lambda *a, **k: print(*a, **k, file=out)
    p("\n--- Model Comparison (averaged across envs & horizons) ---\n")
    model_scores: dict[str, list[float]] = defaultdict(list)
    model_regrets: dict[str, list[float]] = defaultdict(list)
    model_expl: dict[str, list[float]] = defaultdict(list)

    for run in results:
        for ep in run["episodes"]:
            m = ep["metrics"]
            model_scores[run["model"]].append(m["mean_reward"])
            model_regrets[run["model"]].append(m["total_regret"])
            model_expl[run["model"]].append(m["final_exploration_ratio"])

    p(f"{'Model':<45} {'MeanReward':>11} {'MeanRegret':>11} {'ExplRatio':>10}")
    p("-" * 80)
    for model in sorted(model_scores):
        p(
            f"{model:<45} "
            f"{np.mean(model_scores[model]):>11.4f} "
            f"{np.mean(model_regrets[model]):>11.2f} "
            f"{np.mean(model_expl[model]):>10.3f}"
        )


def compare_temperatures(results: list[dict], out=None):
    """Show how temperature affects exploration behavior."""
    p = lambda *a, **k: print(*a, **k, file=out)
    p("\n--- Temperature Effects ---\n")
    temp_data: dict[float, dict[str, list]] = defaultdict(lambda: defaultdict(list))

    for run in results:
        temp = run["temperature"]
        for ep in run["episodes"]:
            m = ep["metrics"]
            temp_data[temp]["reward"].append(m["mean_reward"])
            temp_data[temp]["regret"].append(m["total_regret"])
            temp_data[temp]["exploration"].append(m["final_exploration_ratio"])

    p(f"{'Temp':>6} {'MeanReward':>11} {'MeanRegret':>11} {'ExplRatio':>10} {'N':>5}")
    p("-" * 50)
    for temp in sorted(temp_data):
        d = temp_data[temp]
        n = len(d["reward"])
        p(
            f"{temp:>6.2f} "
            f"{np.mean(d['reward']):>11.4f} "
            f"{np.mean(d['regret']):>11.2f} "
            f"{np.mean(d['exploration']):>10.3f} "
            f"{n:>5}"
        )


def regret_curves(results: list[dict], out=None):
    """Print average regret curves (text-based)."""
    p = lambda *a, **k: print(*a, **k, file=out)
    p("\n--- Regret Curves (avg cumulative regret at each step) ---\n")

    for run in results:
        model = run["model"]
        env = run["environment"]
        temp = run["temperature"]

        by_horizon: dict[int, list] = defaultdict(list)
        for ep in run["episodes"]:
            by_horizon[ep["horizon"]].append(ep["metrics"]["cumulative_regret_curve"])

        for horizon, curves in sorted(by_horizon.items()):
            min_len = min(len(c) for c in curves)
            avg_curve = np.mean([c[:min_len] for c in curves], axis=0)

            indices = np.linspace(0, min_len - 1, min(10, min_len), dtype=int)
            points = " -> ".join(f"r{i+1}:{avg_curve[i]:.2f}" for i in indices)
            p(f"{model} | {env} | t={temp} | h={horizon}: {points}")


def adaptation_analysis(results: list[dict], out=None):
    """Analyze adaptation speed in non-stationary environments."""
    p = lambda *a, **k: print(*a, **k, file=out)
    p("\n--- Adaptation Speed (non-stationary envs only) ---\n")
    found = False

    for run in results:
        for ep in run["episodes"]:
            m = ep["metrics"]
            if m["adaptation_speed"] is not None:
                found = True
                p(
                    f"{run['model']} | {run['environment']} | t={run['temperature']} | "
                    f"h={ep['horizon']}: avg {m['adaptation_speed']:.1f} steps to adapt "
                    f"({len(m['adaptation_events'])} shift events)"
                )

    if not found:
        p("No adaptation events detected (need non-stationary envs with enough horizon).")


def run_all_analyses(results: list[dict], out=None):
    """Run all analyses, writing to the given output stream."""
    summary_table(results, out=out)
    compare_models(results, out=out)
    compare_temperatures(results, out=out)
    regret_curves(results, out=out)
    adaptation_analysis(results, out=out)


def save_analysis(results: list[dict], run_dir: Path) -> Path:
    """Run all analyses and save to analysis.txt in the run directory."""
    filepath = run_dir / "analysis.txt"
    with open(filepath, "w") as f:
        run_all_analyses(results, out=f)
    # also print to stdout
    run_all_analyses(results)
    return filepath


def load_results(path: str) -> list[dict]:
    """Load results. Handle both single runs and sweep arrays."""
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    return [data]


def main():
    """CLI entry point for analyzing results."""
    parser = argparse.ArgumentParser(description="Analyze ee-bench results")
    parser.add_argument("results_file", help="Path to results JSON")
    parser.add_argument("--compare-models", action="store_true")
    parser.add_argument("--compare-temps", action="store_true")
    parser.add_argument("--regret-curves", action="store_true")
    parser.add_argument("--adaptation", action="store_true")
    parser.add_argument("--all", action="store_true", help="Run all analyses")
    parser.add_argument("--output-dir", default=None,
                        help="Directory to save analysis.txt and graphs (defaults to same dir as results)")
    args = parser.parse_args()

    results = load_results(args.results_file)
    do_all = args.all or not any([args.compare_models, args.compare_temps, args.regret_curves, args.adaptation])

    out_dir = Path(args.output_dir) if args.output_dir else Path(args.results_file).parent

    if do_all:
        save_analysis(results, out_dir)
        try:
            from .graphs import generate_graphs
            graphs = generate_graphs(results, out_dir)
            print(f"\nGraphs saved: {len(graphs)} files in {out_dir / 'graphs'}")
        except ImportError:
            print("\nInstall matplotlib for graph generation: pip install ee-bench[analysis]")
    else:
        summary_table(results)
        if args.compare_models:
            compare_models(results)
        if args.compare_temps:
            compare_temperatures(results)
        if args.regret_curves:
            regret_curves(results)
        if args.adaptation:
            adaptation_analysis(results)
