"""Experiment runner — orchestrates environment + model interaction."""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.text import Text

from .config import ExperimentConfig, SweepConfig
from .environments import ALL_ENVIRONMENTS
from .environments.base import Environment
from .metrics.core import EpisodeMetrics, compute_all_metrics
from .providers.openrouter import OpenRouterProvider

console = Console()

# Verbosity levels
QUIET = 0      # no output except errors
PROGRESS = 1   # progress bars only (default)
ACTIONS = 2    # show each action + reward
DEBUG = 3      # show full prompts + raw LLM responses

ENV_MAP = {env.name: env for env in ALL_ENVIRONMENTS}

_verbosity = PROGRESS


def set_verbosity(level: int):
    global _verbosity
    _verbosity = level


def _log(level: int, msg: str, **kwargs):
    if _verbosity >= level:
        console.print(msg, **kwargs)


def _build_messages(env: Environment, include_history: bool = True) -> list[dict[str, str]]:
    """Build the message list for a single turn."""
    messages = [{"role": "system", "content": env.get_system_prompt()}]
    if include_history and env.history:
        messages.append({"role": "user", "content": env.get_action_prompt()})
    else:
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

    if _verbosity >= DEBUG:
        _log(DEBUG, Panel(env.get_system_prompt(), title="[bold]System Prompt[/bold]", border_style="blue"))

    for step in range(horizon):
        messages = _build_messages(env)
        optimal_rewards.append(env.optimal_reward())

        if _verbosity >= DEBUG:
            _log(DEBUG, f"\n[dim]── Step {step+1}/{horizon} ──[/dim]")
            _log(DEBUG, Panel(messages[-1]["content"], title="[bold]Action Prompt[/bold]", border_style="cyan"))

        # get LLM response with retries for parse failures
        action = None
        for attempt in range(max_retries):
            try:
                raw = provider.complete(messages, model=model, temperature=temperature)
            except Exception as e:
                _log(PROGRESS, f"  [red]API error: {e}[/red]")
                time.sleep(2)
                continue

            if _verbosity >= DEBUG:
                _log(DEBUG, f"  [magenta]Raw LLM response:[/magenta] {raw!r}")

            action = env.parse_action(raw)
            if action is not None:
                break

            if _verbosity >= DEBUG:
                _log(DEBUG, f"  [yellow]Parse failed (attempt {attempt+1}), retrying...[/yellow]")

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
            _log(PROGRESS, f"  [yellow]Step {step+1}: parse failed, random fallback → {action}[/yellow]")

        result = env.step(action)

        if _verbosity >= ACTIONS:
            reward_color = "green" if result.reward > 0.6 else "yellow" if result.reward > 0.3 else "red"
            _log(ACTIONS, f"  [{reward_color}]Step {step+1:>3}[/{reward_color}]: {action:<25} → reward={result.reward:.3f}  │ {result.feedback}")

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

    _log(PROGRESS, f"[bold]Model:[/bold] {config.model}")
    _log(PROGRESS, f"[bold]Environment:[/bold] {config.environment} ({'stationary' if env_cls.is_stationary else 'non-stationary'})")
    _log(PROGRESS, f"[bold]Temperature:[/bold] {config.temperature}")
    _log(PROGRESS, f"[bold]Horizons:[/bold] {config.horizons}")
    _log(PROGRESS, f"[bold]Repetitions:[/bold] {config.repetitions}")
    _log(PROGRESS, "")

    use_progress_bar = _verbosity == PROGRESS

    if use_progress_bar:
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
    else:
        for horizon in config.horizons:
            for rep in range(config.repetitions):
                _log(ACTIONS, f"\n[bold]━━━ Horizon {horizon} │ Rep {rep+1}/{config.repetitions} ━━━[/bold]")
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
                _log(ACTIONS, f"  [dim]total_reward={metrics.total_reward:.3f}  regret={metrics.total_regret:.3f}  expl_ratio={metrics.final_exploration_ratio:.3f}[/dim]")

    provider.close()
    return results


def run_sweep(config: SweepConfig) -> list[dict[str, Any]]:
    """Run a full parameter sweep."""
    provider = OpenRouterProvider(config.api_key, config.base_url)
    all_results = []

    envs = config.environments or list(ENV_MAP.keys())
    models = config.models

    total = len(models) * len(envs) * len(config.temperatures) * len(config.horizons) * config.repetitions
    _log(PROGRESS, f"[bold]Sweep:[/bold] {len(models)} models × {len(envs)} envs × {len(config.temperatures)} temps × {len(config.horizons)} horizons × {config.repetitions} reps = {total} episodes")

    use_progress_bar = _verbosity == PROGRESS

    if use_progress_bar:
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
    else:
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
                            _log(ACTIONS, f"\n[bold]━━━ {model} / {env_name} / t={temp} / h={horizon} / r={rep+1} ━━━[/bold]")
                            seed = config.seed + rep * 1000 + horizon
                            env = env_cls(seed=seed)
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
                            _log(ACTIONS, f"  [dim]reward={metrics.total_reward:.3f}  regret={metrics.total_regret:.3f}[/dim]")
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

    _log(PROGRESS, f"[green]Results saved to {filepath}[/green]")
    return filepath
