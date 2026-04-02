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


class EpisodeFailure(Exception):
    """Raised when an episode cannot continue (e.g. model won't produce valid actions)."""

    pass


# Verbosity levels
QUIET = 0  # no output except errors
PROGRESS = 1  # progress bars only (default)
ACTIONS = 2  # show each action + reward
DEBUG = 3  # show full prompts + raw LLM responses

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
) -> tuple[list[dict[str, Any]], list[float], list[dict[str, Any]]]:
    """Run a single episode: horizon steps of interaction.

    Returns (history, optimal_rewards, conversation_log).
    """
    optimal_rewards = []
    conversation_log: list[dict[str, Any]] = []

    if _verbosity >= DEBUG:
        _log(
            DEBUG,
            Panel(env.get_system_prompt(), title="[bold]System Prompt[/bold]", border_style="blue"),
        )

    for step in range(horizon):
        messages = _build_messages(env)
        optimal_rewards.append(env.optimal_reward())

        step_log: dict[str, Any] = {
            "step": step + 1,
            "messages_sent": messages,
            "attempts": [],
        }

        if _verbosity >= DEBUG:
            _log(DEBUG, f"\n[dim]── Step {step + 1}/{horizon} ──[/dim]")
            _log(
                DEBUG,
                Panel(
                    messages[-1]["content"], title="[bold]Action Prompt[/bold]", border_style="cyan"
                ),
            )

        # get LLM response with retries for parse failures
        action = None
        for attempt in range(max_retries):
            attempt_log: dict[str, Any] = {"attempt": attempt + 1}
            try:
                raw = provider.complete(messages, model=model, temperature=temperature)
                attempt_log["raw_response"] = raw
            except Exception as e:
                attempt_log["error"] = str(e)
                _log(PROGRESS, f"  [red]API error: {e}[/red]")
                step_log["attempts"].append(attempt_log)
                time.sleep(2)
                continue

            if _verbosity >= DEBUG:
                _log(DEBUG, f"  [magenta]Raw LLM response:[/magenta] {raw!r}")

            action = env.parse_action(raw)
            attempt_log["parsed_action"] = action
            step_log["attempts"].append(attempt_log)

            if action is not None:
                break

            if _verbosity >= DEBUG:
                _log(DEBUG, f"  [yellow]Parse failed (attempt {attempt + 1}), retrying...[/yellow]")

            # parse failed — add a nudge and retry
            valid = env.valid_actions()
            if len(valid) <= 20:
                hint = f"Please respond with exactly one of: {', '.join(valid)}"
            else:
                hint = "Please respond with only the requested format. No explanation."
            messages.append({"role": "assistant", "content": raw})
            messages.append(
                {"role": "user", "content": f"I couldn't understand your response. {hint}"}
            )

        if action is None:
            step_log["outcome"] = "failure"
            conversation_log.append(step_log)
            raise EpisodeFailure(
                f"Step {step + 1}/{horizon}: model failed to produce a valid action after {max_retries} attempts"
            )

        result = env.step(action)
        step_log["outcome"] = "success"
        step_log["action"] = action
        step_log["reward"] = result.reward
        step_log["feedback"] = result.feedback
        conversation_log.append(step_log)

        if _verbosity >= ACTIONS:
            reward_color = (
                "green" if result.reward > 0.6 else "yellow" if result.reward > 0.3 else "red"
            )
            _log(
                ACTIONS,
                f"  [{reward_color}]Step {step + 1:>3}[/{reward_color}]: {action:<25} → reward={result.reward:.3f}  │ {result.feedback}",
            )

    return env.history, optimal_rewards, conversation_log


def _run_episode_safe(
    env_cls,
    seed: int,
    provider: OpenRouterProvider,
    model: str,
    temperature: float,
    horizon: int,
    episode_retries: int = 3,
) -> dict[str, Any] | None:
    """Run an episode with retries on failure. Returns episode dict or None if unrecoverable."""
    for attempt in range(episode_retries):
        env = env_cls(seed=seed)
        try:
            history, optimal_rewards, conversation_log = run_episode(
                env, provider, model, temperature, horizon
            )
        except EpisodeFailure as e:
            _log(
                PROGRESS,
                f"  [yellow]Episode failed (attempt {attempt + 1}/{episode_retries}): {e}[/yellow]",
            )
            if attempt < episode_retries - 1:
                time.sleep(5 * (attempt + 1))
            continue
        metrics = compute_all_metrics(
            history,
            optimal_rewards,
            is_stationary=env.is_stationary,
        )
        return {
            "metrics": asdict(metrics),
            "history": history,
            "conversation_log": conversation_log,
        }
    _log(PROGRESS, f"  [red]Episode skipped after {episode_retries} failures[/red]")
    return None


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
        raise ValueError(
            f"Unknown environment: {config.environment}. Available: {list(ENV_MAP.keys())}"
        )

    _log(PROGRESS, f"[bold]Model:[/bold] {config.model}")
    _log(
        PROGRESS,
        f"[bold]Environment:[/bold] {config.environment} ({'stationary' if env_cls.is_stationary else 'non-stationary'})",
    )
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
                    ep = _run_episode_safe(
                        env_cls, seed, provider, config.model, config.temperature, horizon
                    )
                    if ep is not None:
                        ep.update({"horizon": horizon, "repetition": rep, "seed": seed})
                        results["episodes"].append(ep)
                    progress.advance(task)
    else:
        for horizon in config.horizons:
            for rep in range(config.repetitions):
                _log(
                    ACTIONS,
                    f"\n[bold]━━━ Horizon {horizon} │ Rep {rep + 1}/{config.repetitions} ━━━[/bold]",
                )
                seed = config.seed + rep * 1000 + horizon
                ep = _run_episode_safe(
                    env_cls, seed, provider, config.model, config.temperature, horizon
                )
                if ep is not None:
                    ep.update({"horizon": horizon, "repetition": rep, "seed": seed})
                    results["episodes"].append(ep)
                    m = ep["metrics"]
                    _log(
                        ACTIONS,
                        f"  [dim]total_reward={m['total_reward']:.3f}  regret={m['total_regret']:.3f}  expl_ratio={m['final_exploration_ratio']:.3f}[/dim]",
                    )

    provider.close()
    return results


def _find_combo(
    results: list[dict[str, Any]], model: str, env_name: str, temp: float
) -> dict[str, Any] | None:
    """Find an existing combo result dict, or None."""
    for r in results:
        if r["model"] == model and r["environment"] == env_name and r["temperature"] == temp:
            return r
    return None


def _episode_done(combo: dict[str, Any], horizon: int, rep: int) -> bool:
    """Check if a specific episode already exists in a combo."""
    return any(
        ep["horizon"] == horizon and ep["repetition"] == rep for ep in combo["episodes"]
    )


def _flush_results(all_results: list[dict[str, Any]], run_dir: Path):
    """Atomically write current results to disk."""
    filepath = run_dir / "results.json"
    tmp = filepath.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    tmp.rename(filepath)


def run_sweep(
    config: SweepConfig,
    run_dir: Path | None = None,
    existing_results: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], Path]:
    """Run a full parameter sweep with incremental saves.

    Args:
        config: Sweep configuration.
        run_dir: Directory to save results into. Created if None.
        existing_results: Previously completed results (for --resume).

    Returns:
        (all_results, run_dir) — includes both existing and new results.
    """
    provider = OpenRouterProvider(config.api_key, config.base_url)

    # Set up run directory
    if run_dir is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        run_dir = Path(config.results_dir) / f"sweep_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    all_results = list(existing_results) if existing_results else []

    envs = config.environments or list(ENV_MAP.keys())
    models = config.models

    # Count total and already-done episodes for progress
    total_episodes = len(models) * len(envs) * len(config.temperatures) * len(config.horizons) * config.repetitions
    skipped_episodes = sum(len(r["episodes"]) for r in all_results)
    remaining = total_episodes - skipped_episodes

    if skipped_episodes:
        _log(
            PROGRESS,
            f"[bold]Resuming:[/bold] {skipped_episodes}/{total_episodes} episodes already done, {remaining} remaining",
        )
    _log(
        PROGRESS,
        f"[bold]Sweep:[/bold] {len(models)} models × {len(envs)} envs × {len(config.temperatures)} temps × {len(config.horizons)} horizons × {config.repetitions} reps = {total_episodes} episodes",
    )

    use_progress_bar = _verbosity == PROGRESS

    if use_progress_bar:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            main_task = progress.add_task("Sweep", total=remaining)
            for model in models:
                for env_name in envs:
                    for temp in config.temperatures:
                        combo = _find_combo(all_results, model, env_name, temp)
                        if combo is None:
                            combo = {"model": model, "environment": env_name, "temperature": temp, "episodes": []}
                            all_results.append(combo)
                        env_cls = ENV_MAP[env_name]
                        for horizon in config.horizons:
                            for rep in range(config.repetitions):
                                if _episode_done(combo, horizon, rep):
                                    continue
                                seed = config.seed + rep * 1000 + horizon
                                progress.update(
                                    main_task,
                                    description=f"{model} / {env_name} / t={temp} / h={horizon} / r={rep + 1}",
                                )
                                ep = _run_episode_safe(
                                    env_cls, seed, provider, model, temp, horizon
                                )
                                if ep is not None:
                                    ep.update({"horizon": horizon, "repetition": rep, "seed": seed})
                                    combo["episodes"].append(ep)
                                _flush_results(all_results, run_dir)
                                progress.advance(main_task)
    else:
        for model in models:
            for env_name in envs:
                for temp in config.temperatures:
                    combo = _find_combo(all_results, model, env_name, temp)
                    if combo is None:
                        combo = {"model": model, "environment": env_name, "temperature": temp, "episodes": []}
                        all_results.append(combo)
                    env_cls = ENV_MAP[env_name]
                    combo_had_work = False
                    for horizon in config.horizons:
                        for rep in range(config.repetitions):
                            if _episode_done(combo, horizon, rep):
                                _log(
                                    ACTIONS,
                                    f"\n[dim]Skipping {model} / {env_name} / t={temp} / h={horizon} / r={rep + 1} (already done)[/dim]",
                                )
                                continue
                            combo_had_work = True
                            _log(
                                ACTIONS,
                                f"\n[bold]━━━ {model} / {env_name} / t={temp} / h={horizon} / r={rep + 1} ━━━[/bold]",
                            )
                            seed = config.seed + rep * 1000 + horizon
                            ep = _run_episode_safe(env_cls, seed, provider, model, temp, horizon)
                            if ep is not None:
                                ep.update({"horizon": horizon, "repetition": rep, "seed": seed})
                                combo["episodes"].append(ep)
                                _log(
                                    ACTIONS,
                                    f"  [dim]reward={ep['metrics']['total_reward']:.3f}  regret={ep['metrics']['total_regret']:.3f}[/dim]",
                                )
                            _flush_results(all_results, run_dir)
                    if not combo_had_work:
                        _log(
                            ACTIONS,
                            f"\n[dim]Skipping {model} / {env_name} / t={temp} (fully done)[/dim]",
                        )

    provider.close()
    return all_results, run_dir


def save_results(results: dict | list, output_dir: str, name: str | None = None) -> Path:
    """Save results to a timestamped run directory.

    Creates: <output_dir>/<name>_<timestamp>/results.json
    Returns the run directory path.
    """
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    if name is None:
        name = "run"

    run_dir = Path(output_dir) / f"{name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    filepath = run_dir / "results.json"
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2, default=str)

    _log(PROGRESS, f"[green]Results saved to {filepath}[/green]")
    return run_dir
