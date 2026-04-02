"""Experiment runner — orchestrates environment + model interaction."""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from .config import ExperimentConfig, SweepConfig
from .environments import ALL_ENVIRONMENTS
from .environments.base import Environment
from .metrics.core import EpisodeMetrics, compute_all_metrics
from .providers.openrouter import OpenRouterProvider

console = Console()

ENV_MAP = {env.name: env for env in ALL_ENVIRONMENTS}


def _build_messages(env: Environment, include_history: bool = True) -> list[dict[str, str]]:
    """Build the message list for a single turn."""
    messages = [{"role": "system", "content": env.get_system_prompt()}]
    if include_history and env.history:
        # include history as a user message for context
        messages.append({"role": "user", "content": env.get_action_prompt()})
    else:
        actions_text = ", ".join(env.valid_actions()[:20])  # don't list all for search envs
        first_prompt = env.get_action_prompt()
        messages.append({"role": "user", "content": first_prompt})
    return messages


def run_episode(
    env: Environment,
    provider: OpenRouterProvider,
    model: str,
    temperature: float,
    horizon: int,
    max_retries: int = 3,
) -> tuple[list[dict[str, Any]], list[float]]:
    """Run a single episode: horizon steps of interaction.

    Returns (history, optimal_rewards).
    """
    optimal_rewards = []

    for step in range(horizon):
        messages = _build_messages(env)
        optimal_rewards.append(env.optimal_reward())

        # get LLM response with retries for parse failures
        action = None
        for attempt in range(max_retries):
            try:
                raw = provider.complete(messages, model=model, temperature=temperature)
            except Exception as e:
                console.print(f"  [red]API error: {e}[/red]")
                time.sleep(2)
                continue

            action = env.parse_action(raw)
            if action is not None:
                break

            # parse failed — add a nudge and retry
            valid = env.valid_actions()
            if len(valid) <= 20:
                hint = f"Please respond with exactly one of: {', '.join(valid)}"
            else:
                hint = "Please respond with only the requested format. No explanation."
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content": f"I couldn't understand your response. {hint}"})

        if action is None:
            # fallback: pick a random valid action
            action = env.rng.choice(env.valid_actions())
            console.print(f"  [yellow]Step {step+1}: parse failed, random fallback → {action}[/yellow]")

        result = env.step(action)

    return env.history, optimal_rewards


def run_single(config: ExperimentConfig) -> dict[str, Any]:
    """Run a single experiment configuration (one model, one env, all horizons)."""
    provider = OpenRouterProvider(config.api_key, config.base_url)
    results: dict[str, Any] = {
        "model": config.model,
        "environment": config.environment,
        "temperature": config.temperature,
        "episodes": [],
    }

    env_cls = ENV_MAP.get(config.environment)
    if env_cls is None:
        raise ValueError(f"Unknown environment: {config.environment}. Available: {list(ENV_MAP.keys())}")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        for horizon in config.horizons:
            task = progress.add_task(
                f"{config.model} / {config.environment} / h={horizon}",
                total=config.repetitions,
            )
            for rep in range(config.repetitions):
                seed = config.seed + rep * 1000 + horizon
                env = env_cls(seed=seed)
                history, optimal_rewards = run_episode(
                    env, provider, config.model, config.temperature, horizon
                )
                metrics = compute_all_metrics(
                    history, optimal_rewards, is_stationary=env.is_stationary
                )
                results["episodes"].append({
                    "horizon": horizon,
                    "repetition": rep,
                    "seed": seed,
                    "metrics": asdict(metrics),
                    "history": history,
                })
                progress.advance(task)

    provider.close()
    return results


def run_sweep(config: SweepConfig) -> list[dict[str, Any]]:
    """Run a full parameter sweep."""
    provider = OpenRouterProvider(config.api_key, config.base_url)
    all_results = []

    envs = config.environments or list(ENV_MAP.keys())
    models = config.models

    total = len(models) * len(envs) * len(config.temperatures) * len(config.horizons) * config.repetitions

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        main_task = progress.add_task("Sweep", total=total)

        for model in models:
            for env_name in envs:
                for temp in config.temperatures:
                    run_result = {
                        "model": model,
                        "environment": env_name,
                        "temperature": temp,
                        "episodes": [],
                    }
                    env_cls = ENV_MAP[env_name]

                    for horizon in config.horizons:
                        for rep in range(config.repetitions):
                            seed = config.seed + rep * 1000 + horizon
                            env = env_cls(seed=seed)
                            progress.update(
                                main_task,
                                description=f"{model} / {env_name} / t={temp} / h={horizon} / r={rep+1}",
                            )
                            history, optimal_rewards = run_episode(
                                env, provider, model, temp, horizon
                            )
                            metrics = compute_all_metrics(
                                history, optimal_rewards, is_stationary=env.is_stationary,
                            )
                            run_result["episodes"].append({
                                "horizon": horizon,
                                "repetition": rep,
                                "seed": seed,
                                "metrics": asdict(metrics),
                                "history": history,
                            })
                            progress.advance(main_task)

                    all_results.append(run_result)

    provider.close()
    return all_results


def save_results(results: dict | list, output_dir: str, name: str | None = None):
    """Save results to JSON."""
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)

    if name is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        name = f"run_{timestamp}"

    filepath = path / f"{name}.json"
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2, default=str)

    console.print(f"[green]Results saved to {filepath}[/green]")
    return filepath
