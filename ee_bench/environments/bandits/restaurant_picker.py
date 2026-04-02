"""Restaurant Picker environment - stationary multi-armed bandit with contextual flavor."""

from __future__ import annotations

import numpy as np

from ..base import Environment, EnvironmentResult
from ...prompts import load_prompt


class RestaurantPicker(Environment):
    """5 restaurants — tourist trap with flashy name, hidden gem elsewhere."""

    name = "restaurant_picker"
    description = "5 restaurants in a new city — find your favorite"
    is_stationary = True

    RESTAURANTS = [
        "Mama Rosa's Trattoria",
        "The Gilded Fork",
        "Szechuan Alley",
        "Neptune's Catch",
        "Burger Philosophy",
    ]

    def __init__(self, seed: int = 42):
        super().__init__(seed)
        self._prompts = load_prompt(self.name)
        self._means = self.rng.uniform(3.0, 8.5, size=5)
        gem = self.rng.integers(0, 5)
        self._means[gem] = 8.5 + self.rng.uniform(0, 1.0)
        self._means[1] = 3.5  # tourist trap
        self._stds = self.rng.uniform(0.5, 1.5, size=5)

    def get_system_prompt(self) -> str:
        restaurant_list = "\n".join(f"  - {r}" for r in self.RESTAURANTS)
        prompt = self._prompts.system.format(restaurant_list=restaurant_list)
        return self._append_format_instruction(prompt)

    def get_action_prompt(self) -> str:
        return self._prompts.action.format(
            history=self.format_history(last_n=20),
            round=self.step_count + 1,
        )

    def valid_actions(self) -> list[str]:
        return list(self.RESTAURANTS)

    def step(self, action: str) -> EnvironmentResult:
        idx = self.RESTAURANTS.index(action)
        score = float(np.clip(self.rng.normal(self._means[idx], self._stds[idx]), 1, 10))
        score_int = int(round(score))
        self.step_count += 1

        flavor = self._prompts.pick_flavor(score_int, self.rng)
        summary = f"Ate at {action} → {score_int}/10"
        self.history.append(
            {
                "round": self.step_count,
                "action": action,
                "reward": score / 10,
                "summary": summary,
            }
        )
        feedback = self._prompts.feedback.format(action=action, score=score_int, flavor=flavor)
        return EnvironmentResult(reward=score / 10, feedback=feedback)

    def optimal_reward(self) -> float:
        return float(np.max(self._means)) / 10
