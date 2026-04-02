#!/usr/bin/env python3
"""Analysis scripts for ee-bench results.

Usage:
    python analyze.py results/sweep.json
    python analyze.py results/sweep.json --compare-models
    python analyze.py results/sweep.json --compare-temps
    python analyze.py results/sweep.json --regret-curves
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_results(path: str) -> list[dict]:
    """Load results. Handle both single runs and sweep arrays."""
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    return [data]


def summary_table(results: list[dict]):
    """Print a summary table of all runs."""
    print(f"\n{'Model':<40} {'Env':<25} {'Temp':>5} {'Horizon':>8} {'AvgReward':>10} {'AvgRegret':>10} {'ExplRatio':>10} {'UniqueAct':>10}")
    print("=" * 130)

    for run in results:
        model = run["model"]
        env = run["environment"]
        temp = run["temperature"]

        # group episodes by horizon
        by_horizon: dict[int, list] = defaultdict(list)
        for ep in run["episodes"]:
            by_horizon[ep["horizon"]].append(ep["metrics"])

        for horizon, metrics_list in sorted(by_horizon.items()):
            avg_reward = np.mean([m["mean_reward"] for m in metrics_list])
            avg_regret = np.mean([m["total_regret"] for m in metrics_list])
            avg_expl = np.mean([m["final_exploration_ratio"] for m in metrics_list])
            avg_unique = np.mean([m["unique_actions_tried"] for m in metrics_list])
            print(f"{model:<40} {env:<25} {temp:>5.1f} {horizon:>8} {avg_reward:>10.4f} {avg_regret:>10.2f} {avg_expl:>10.3f} {avg_unique:>10.1f}")


def compare_models(results: list[dict]):
    """Compare models across environments."""
    print("\n--- Model Comparison (averaged across envs & horizons) ---\n")
    model_scores: dict[str, list[float]] = defaultdict(list)
    model_regrets: dict[str, list[float]] = defaultdict(list)
    model_expl: dict[str, list[float]] = defaultdict(list)

    for run in results:
        for ep in run["episodes"]:
            m = ep["metrics"]
            model_scores[run["model"]].append(m["mean_reward"])
            model_regrets[run["model"]].append(m["total_regret"])
            model_expl[run["model"]].append(m["final_exploration_ratio"])

    print(f"{'Model':<45} {'MeanReward':>11} {'MeanRegret':>11} {'ExplRatio':>10}")
    print("-" * 80)
    for model in sorted(model_scores):
        print(
            f"{model:<45} "
            f"{np.mean(model_scores[model]):>11.4f} "
            f"{np.mean(model_regrets[model]):>11.2f} "
            f"{np.mean(model_expl[model]):>10.3f}"
        )


def compare_temperatures(results: list[dict]):
    """Show how temperature affects exploration behavior."""
    print("\n--- Temperature Effects ---\n")
    temp_data: dict[float, dict[str, list]] = defaultdict(lambda: defaultdict(list))

    for run in results:
        temp = run["temperature"]
        for ep in run["episodes"]:
            m = ep["metrics"]
            temp_data[temp]["reward"].append(m["mean_reward"])
            temp_data[temp]["regret"].append(m["total_regret"])
            temp_data[temp]["exploration"].append(m["final_exploration_ratio"])

    print(f"{'Temp':>6} {'MeanReward':>11} {'MeanRegret':>11} {'ExplRatio':>10} {'N':>5}")
    print("-" * 50)
    for temp in sorted(temp_data):
        d = temp_data[temp]
        n = len(d["reward"])
        print(
            f"{temp:>6.2f} "
            f"{np.mean(d['reward']):>11.4f} "
            f"{np.mean(d['regret']):>11.2f} "
            f"{np.mean(d['exploration']):>10.3f} "
            f"{n:>5}"
        )


def regret_curves(results: list[dict]):
    """Print average regret curves (text-based, for quick analysis)."""
    print("\n--- Regret Curves (avg cumulative regret at each step) ---\n")

    for run in results:
        model = run["model"]
        env = run["environment"]
        temp = run["temperature"]

        by_horizon: dict[int, list] = defaultdict(list)
        for ep in run["episodes"]:
            by_horizon[ep["horizon"]].append(ep["metrics"]["cumulative_regret_curve"])

        for horizon, curves in sorted(by_horizon.items()):
            # average curves (they may have slightly different lengths due to parse failures)
            min_len = min(len(c) for c in curves)
            avg_curve = np.mean([c[:min_len] for c in curves], axis=0)

            # show at 10 evenly spaced points
            indices = np.linspace(0, min_len - 1, min(10, min_len), dtype=int)
            points = " → ".join(f"r{i+1}:{avg_curve[i]:.2f}" for i in indices)
            print(f"{model} | {env} | t={temp} | h={horizon}: {points}")


def adaptation_analysis(results: list[dict]):
    """Analyze adaptation speed in non-stationary environments."""
    print("\n--- Adaptation Speed (non-stationary envs only) ---\n")
    found = False

    for run in results:
        for ep in run["episodes"]:
            m = ep["metrics"]
            if m["adaptation_speed"] is not None:
                found = True
                print(
                    f"{run['model']} | {run['environment']} | t={run['temperature']} | "
                    f"h={ep['horizon']}: avg {m['adaptation_speed']:.1f} steps to adapt "
                    f"({len(m['adaptation_events'])} shift events)"
                )

    if not found:
        print("No adaptation events detected (need non-stationary envs with enough horizon).")


def main():
    parser = argparse.ArgumentParser(description="Analyze ee-bench results")
    parser.add_argument("results_file", help="Path to results JSON")
    parser.add_argument("--compare-models", action="store_true")
    parser.add_argument("--compare-temps", action="store_true")
    parser.add_argument("--regret-curves", action="store_true")
    parser.add_argument("--adaptation", action="store_true")
    parser.add_argument("--all", action="store_true", help="Run all analyses")
    args = parser.parse_args()

    results = load_results(args.results_file)
    run_all = args.all or not any([args.compare_models, args.compare_temps, args.regret_curves, args.adaptation])

    summary_table(results)

    if args.compare_models or run_all:
        compare_models(results)
    if args.compare_temps or run_all:
        compare_temperatures(results)
    if args.regret_curves or run_all:
        regret_curves(results)
    if args.adaptation or run_all:
        adaptation_analysis(results)


if __name__ == "__main__":
    main()
