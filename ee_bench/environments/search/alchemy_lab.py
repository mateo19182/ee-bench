"""Alchemy Lab environment - combinatorial search for optimal potion recipe."""

from __future__ import annotations

import re

import numpy as np

from ..base import Environment, EnvironmentResult
from ...prompts import load_prompt


class AlchemyLab(Environment):
    """Mix 3 ingredients to find the optimal potion recipe."""

    name = "alchemy_lab"
    description = "Mix 3 ingredients to find the optimal potion recipe"
    is_stationary = True

    INGREDIENTS = ["Dragon's Breath", "Moon Dust", "Serpent Oil"]

    def __init__(self, seed: int = 42):
        super().__init__(seed)
        self._prompts = load_prompt(self.name)
        self._optimal = self.rng.integers(1, 9, size=3).astype(float)
        self._width = self.rng.uniform(2.0, 4.0, size=3)

    def _potency(self, amounts: tuple[int, int, int]) -> float:
        diff = np.array(amounts, dtype=float) - self._optimal
        exponent = -np.sum((diff / self._width) ** 2)
        return float(np.exp(exponent))

    def get_system_prompt(self) -> str:
        prompt = self._prompts.system.format(
            ingredient_1=self.INGREDIENTS[0],
            ingredient_2=self.INGREDIENTS[1],
            ingredient_3=self.INGREDIENTS[2],
        )
        return self._append_format_instruction(prompt)

    def get_action_prompt(self) -> str:
        return self._prompts.action.format(
            history=self.format_history(last_n=20),
            round=self.step_count + 1,
        )

    def valid_actions(self) -> list[str]:
        return [f"{a},{b},{c}" for a in range(10) for b in range(10) for c in range(10)]

    def parse_action(self, raw: str) -> str | None:
        nums = re.findall(r"(\d+)", raw)
        if len(nums) >= 3:
            a, b, c = int(nums[0]), int(nums[1]), int(nums[2])
            if all(0 <= x <= 9 for x in (a, b, c)):
                return f"{a},{b},{c}"
        return None

    def step(self, action: str) -> EnvironmentResult:
        amounts = tuple(int(x) for x in action.split(","))
        self.step_count += 1

        potency = self._potency(amounts)
        pct = int(round(potency * 100))
        reward = potency

        flavor = self._prompts.pick_flavor(pct, self.rng)
        recipe = f"{self.INGREDIENTS[0]}={amounts[0]}, {self.INGREDIENTS[1]}={amounts[1]}, {self.INGREDIENTS[2]}={amounts[2]}"
        summary = f"Mixed ({action}) → potency {pct}%"
        self.history.append(
            {
                "round": self.step_count,
                "action": action,
                "reward": reward,
                "summary": summary,
            }
        )

        feedback = self._prompts.feedback.format(recipe=recipe, potency=pct, flavor=flavor)
        return EnvironmentResult(reward=reward, feedback=feedback)

    def optimal_reward(self) -> float:
        return 1.0
