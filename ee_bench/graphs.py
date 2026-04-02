"""Graph generation for ee-bench run results."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np


def generate_graphs(results: list[dict], run_dir: Path) -> list[Path]:
    """Generate all graphs and save to run_dir. Returns list of created files."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if isinstance(results, dict):
        results = [results]

    graphs_dir = run_dir / "graphs"
    graphs_dir.mkdir(exist_ok=True)

    created = []
    created += _plot_regret_curves(results, graphs_dir, plt)
    created += _plot_model_comparison(results, graphs_dir, plt)
    created += _plot_exploration_over_time(results, graphs_dir, plt)
    created += _plot_temperature_effects(results, graphs_dir, plt)
    return created


def _plot_regret_curves(results: list[dict], out: Path, plt) -> list[Path]:
    """Cumulative regret curves per model/env/horizon, averaged across reps."""
    created = []

    # Group: (env, horizon) -> {model: [curves]}
    grouped: dict[tuple, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for run in results:
        for ep in run["episodes"]:
            key = (run["environment"], ep["horizon"])
            grouped[key][run["model"]].append(ep["metrics"]["cumulative_regret_curve"])

    for (env, horizon), model_curves in sorted(grouped.items()):
        fig, ax = plt.subplots(figsize=(10, 6))
        for model, curves in sorted(model_curves.items()):
            min_len = min(len(c) for c in curves)
            avg = np.mean([c[:min_len] for c in curves], axis=0)
            steps = np.arange(1, min_len + 1)
            label = model.split("/")[-1] if "/" in model else model
            ax.plot(steps, avg, label=label, linewidth=2)

        ax.set_xlabel("Step")
        ax.set_ylabel("Cumulative Regret")
        ax.set_title(f"Cumulative Regret — {env} (horizon={horizon})")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        fname = out / f"regret_{env}_h{horizon}.png"
        fig.savefig(fname, dpi=150)
        plt.close(fig)
        created.append(fname)

    return created


def _plot_model_comparison(results: list[dict], out: Path, plt) -> list[Path]:
    """Bar chart comparing models on mean reward and mean regret."""
    model_rewards: dict[str, list[float]] = defaultdict(list)
    model_regrets: dict[str, list[float]] = defaultdict(list)

    for run in results:
        for ep in run["episodes"]:
            m = ep["metrics"]
            model_rewards[run["model"]].append(m["mean_reward"])
            model_regrets[run["model"]].append(m["total_regret"])

    if len(model_rewards) < 2:
        return []

    models = sorted(model_rewards.keys())
    labels = [m.split("/")[-1] if "/" in m else m for m in models]
    avg_rewards = [np.mean(model_rewards[m]) for m in models]
    avg_regrets = [np.mean(model_regrets[m]) for m in models]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    x = np.arange(len(models))
    ax1.barh(x, avg_rewards, color="#4C72B0")
    ax1.set_yticks(x)
    ax1.set_yticklabels(labels, fontsize=9)
    ax1.set_xlabel("Mean Reward")
    ax1.set_title("Mean Reward by Model")
    ax1.invert_yaxis()

    ax2.barh(x, avg_regrets, color="#DD8452")
    ax2.set_yticks(x)
    ax2.set_yticklabels(labels, fontsize=9)
    ax2.set_xlabel("Mean Total Regret")
    ax2.set_title("Mean Total Regret by Model")
    ax2.invert_yaxis()

    fig.tight_layout()
    fname = out / "model_comparison.png"
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    return [fname]


def _plot_exploration_over_time(results: list[dict], out: Path, plt) -> list[Path]:
    """Exploration ratio curves per model/env/horizon."""
    created = []

    grouped: dict[tuple, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for run in results:
        for ep in run["episodes"]:
            key = (run["environment"], ep["horizon"])
            grouped[key][run["model"]].append(ep["metrics"]["exploration_ratio_curve"])

    for (env, horizon), model_curves in sorted(grouped.items()):
        fig, ax = plt.subplots(figsize=(10, 6))
        for model, curves in sorted(model_curves.items()):
            min_len = min(len(c) for c in curves)
            avg = np.mean([c[:min_len] for c in curves], axis=0)
            steps = np.arange(1, min_len + 1)
            label = model.split("/")[-1] if "/" in model else model
            ax.plot(steps, avg, label=label, linewidth=2)

        ax.set_xlabel("Step")
        ax.set_ylabel("Exploration Ratio")
        ax.set_title(f"Exploration Ratio — {env} (horizon={horizon})")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        fname = out / f"exploration_{env}_h{horizon}.png"
        fig.savefig(fname, dpi=150)
        plt.close(fig)
        created.append(fname)

    return created


def _plot_temperature_effects(results: list[dict], out: Path, plt) -> list[Path]:
    """Scatter/line plot of temperature vs reward/regret/exploration."""
    temp_data: dict[float, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for run in results:
        temp = run["temperature"]
        for ep in run["episodes"]:
            m = ep["metrics"]
            temp_data[temp]["reward"].append(m["mean_reward"])
            temp_data[temp]["regret"].append(m["total_regret"])
            temp_data[temp]["exploration"].append(m["final_exploration_ratio"])

    if len(temp_data) < 2:
        return []

    temps = sorted(temp_data.keys())
    avg_reward = [np.mean(temp_data[t]["reward"]) for t in temps]
    avg_regret = [np.mean(temp_data[t]["regret"]) for t in temps]
    avg_expl = [np.mean(temp_data[t]["exploration"]) for t in temps]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))

    ax1.plot(temps, avg_reward, "o-", color="#4C72B0", linewidth=2, markersize=8)
    ax1.set_xlabel("Temperature")
    ax1.set_ylabel("Mean Reward")
    ax1.set_title("Temperature vs Reward")
    ax1.grid(True, alpha=0.3)

    ax2.plot(temps, avg_regret, "o-", color="#DD8452", linewidth=2, markersize=8)
    ax2.set_xlabel("Temperature")
    ax2.set_ylabel("Mean Total Regret")
    ax2.set_title("Temperature vs Regret")
    ax2.grid(True, alpha=0.3)

    ax3.plot(temps, avg_expl, "o-", color="#55A868", linewidth=2, markersize=8)
    ax3.set_xlabel("Temperature")
    ax3.set_ylabel("Exploration Ratio")
    ax3.set_title("Temperature vs Exploration")
    ax3.grid(True, alpha=0.3)

    fig.tight_layout()
    fname = out / "temperature_effects.png"
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    return [fname]
