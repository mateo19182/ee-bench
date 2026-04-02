"""Metrics for evaluating explore/exploit behaviour."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class EpisodeMetrics:
    """Metrics computed for one episode (one run of an environment)."""
    total_reward: float
    total_regret: float
    cumulative_regret_curve: list[float]  # regret at each step
    exploration_ratio_curve: list[float]  # rolling exploration ratio
    adaptation_events: list[dict[str, Any]]  # detected strategy shifts
    unique_actions_tried: int
    total_steps: int

    # summary stats
    mean_reward: float
    final_exploration_ratio: float
    adaptation_speed: float | None  # only for non-stationary envs


def compute_all_metrics(
    history: list[dict[str, Any]],
    optimal_rewards: list[float],
    is_stationary: bool,
    window: int = 10,
) -> EpisodeMetrics:
    """Compute all metrics from an episode's history.

    Args:
        history: list of {"round", "action", "reward", "summary"} dicts
        optimal_rewards: optimal reward for each step
        is_stationary: whether the environment is stationary
        window: rolling window for exploration ratio
    """
    n = len(history)
    if n == 0:
        return EpisodeMetrics(
            total_reward=0, total_regret=0,
            cumulative_regret_curve=[], exploration_ratio_curve=[],
            adaptation_events=[], unique_actions_tried=0, total_steps=0,
            mean_reward=0, final_exploration_ratio=0, adaptation_speed=None,
        )

    rewards = [h["reward"] for h in history]
    actions = [h["action"] for h in history]

    # --- Cumulative regret ---
    regrets = [opt - r for opt, r in zip(optimal_rewards, rewards)]
    cum_regret = list(np.cumsum(regrets))

    # --- Exploration ratio ---
    # An action is "exploratory" if it wasn't the empirical best so far
    exploration_curve = []
    action_rewards: dict[str, list[float]] = {}
    for i, (a, r) in enumerate(zip(actions, rewards)):
        action_rewards.setdefault(a, [])
        action_rewards[a].append(r)

        # best action so far by mean reward
        best_action = max(action_rewards, key=lambda k: np.mean(action_rewards[k]))
        is_exploring = (a != best_action) if len(action_rewards) > 1 else True

        # rolling window
        start = max(0, i - window + 1)
        window_actions = actions[start:i + 1]
        # compute how many unique actions in window vs. just picking the best
        if i >= 1:
            window_best = best_action
            explore_count = sum(1 for wa in window_actions if wa != window_best)
            exploration_curve.append(explore_count / len(window_actions))
        else:
            exploration_curve.append(1.0)  # first action is always exploration

    # --- Adaptation speed (non-stationary only) ---
    adaptation_speed = None
    adaptation_events = []
    if not is_stationary and n > 20:
        # detect points where optimal reward changes significantly
        opt_arr = np.array(optimal_rewards)
        for i in range(1, n):
            if abs(opt_arr[i] - opt_arr[i - 1]) > 0.05:
                # environment shifted — how many steps until the LLM adjusts?
                # "adjusts" = starts picking a different action than before
                pre_action = actions[i - 1] if i > 0 else None
                steps_to_adapt = None
                for j in range(i, min(i + 30, n)):
                    if actions[j] != pre_action:
                        steps_to_adapt = j - i
                        break
                adaptation_events.append({
                    "shift_at": i,
                    "steps_to_adapt": steps_to_adapt,
                })

        speeds = [e["steps_to_adapt"] for e in adaptation_events if e["steps_to_adapt"] is not None]
        adaptation_speed = float(np.mean(speeds)) if speeds else None

    return EpisodeMetrics(
        total_reward=sum(rewards),
        total_regret=cum_regret[-1],
        cumulative_regret_curve=cum_regret,
        exploration_ratio_curve=exploration_curve,
        adaptation_events=adaptation_events,
        unique_actions_tried=len(set(actions)),
        total_steps=n,
        mean_reward=float(np.mean(rewards)),
        final_exploration_ratio=float(np.mean(exploration_curve[-window:])) if exploration_curve else 0,
        adaptation_speed=adaptation_speed,
    )
