"""Search and optimization environments.

These differ from bandits: instead of repeated pulls from distributions,
the LLM navigates a space looking for an optimum with limited probes.
"""

from __future__ import annotations

import math

import numpy as np

from .base import Environment, EnvironmentResult
from ..prompts import load_prompt


# -----------------------------------------------------------------------
# 1. Treasure Hunter  (2D grid search with proximity hints)
# -----------------------------------------------------------------------

class TreasureHunter(Environment):
    """10x10 grid search — dig for treasure using proximity hints."""

    name = "treasure_hunter"
    description = "10x10 grid search — dig for treasure using proximity hints"
    is_stationary = True

    def __init__(self, seed: int = 42):
        super().__init__(seed)
        self._prompts = load_prompt(self.name)
        self._target = (int(self.rng.integers(0, 10)), int(self.rng.integers(0, 10)))
        self._best_distance = float('inf')

    def _distance(self, r: int, c: int) -> float:
        return math.sqrt((r - self._target[0]) ** 2 + (c - self._target[1]) ** 2)

    def _signal_strength(self, r: int, c: int) -> float:
        d = self._distance(r, c)
        max_d = math.sqrt(9**2 + 9**2)
        return max(0, 1.0 - d / max_d)

    def get_system_prompt(self) -> str:
        return self._prompts.system

    def get_action_prompt(self) -> str:
        return self._prompts.action.format(
            history=self.format_history(last_n=20),
            round=self.step_count + 1,
        )

    def valid_actions(self) -> list[str]:
        return [f"{r},{c}" for r in range(10) for c in range(10)]

    def parse_action(self, raw: str) -> str | None:
        import re
        nums = re.findall(r'(\d+)', raw)
        if len(nums) >= 2:
            r, c = int(nums[0]), int(nums[1])
            if 0 <= r <= 9 and 0 <= c <= 9:
                return f"{r},{c}"
        return None

    def step(self, action: str) -> EnvironmentResult:
        r, c = map(int, action.split(","))
        self.step_count += 1

        signal = self._signal_strength(r, c)
        signal_pct = int(round(signal * 100))
        dist = self._distance(r, c)

        if dist < self._best_distance:
            direction = "WARMER"
        elif dist > self._best_distance:
            direction = "COLDER"
        else:
            direction = "SAME"
        self._best_distance = min(self._best_distance, dist)

        found = (r, c) == self._target
        flavor = self._prompts.pick_flavor(signal_pct, self.rng, found=found)

        summary = f"Dug ({r},{c}) → signal {signal_pct}% [{direction}]"
        self.history.append({"round": self.step_count, "action": action, "reward": signal, "summary": summary})

        feedback = self._prompts.feedback.format(
            row=r, col=c, signal=signal_pct, direction=direction, flavor=flavor,
        )
        return EnvironmentResult(reward=signal, feedback=feedback, info={"found": found})

    def optimal_reward(self) -> float:
        return 1.0


# -----------------------------------------------------------------------
# 2. Alchemy Lab  (combinatorial search — mix ingredients)
# -----------------------------------------------------------------------

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
        return self._prompts.system.format(
            ingredient_1=self.INGREDIENTS[0],
            ingredient_2=self.INGREDIENTS[1],
            ingredient_3=self.INGREDIENTS[2],
        )

    def get_action_prompt(self) -> str:
        return self._prompts.action.format(
            history=self.format_history(last_n=20),
            round=self.step_count + 1,
        )

    def valid_actions(self) -> list[str]:
        return [f"{a},{b},{c}" for a in range(10) for b in range(10) for c in range(10)]

    def parse_action(self, raw: str) -> str | None:
        import re
        nums = re.findall(r'(\d+)', raw)
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
        self.history.append({"round": self.step_count, "action": action, "reward": reward, "summary": summary})

        feedback = self._prompts.feedback.format(recipe=recipe, potency=pct, flavor=flavor)
        return EnvironmentResult(reward=reward, feedback=feedback)

    def optimal_reward(self) -> float:
        return 1.0


# -----------------------------------------------------------------------
# 3. Radio Tuner  (1D search with noise)
# -----------------------------------------------------------------------

class RadioTuner(Environment):
    """Tune a radio dial (0-100) to find the hidden station, with a decoy."""

    name = "radio_tuner"
    description = "Tune a radio dial (0-100) to find the hidden station"
    is_stationary = True

    def __init__(self, seed: int = 42):
        super().__init__(seed)
        self._prompts = load_prompt(self.name)
        self._main_freq = float(self.rng.uniform(15, 85))
        self._main_strength = 1.0
        self._main_width = float(self.rng.uniform(5, 12))
        self._decoy_freq = float(self._main_freq + self.rng.choice([-1, 1]) * self.rng.uniform(25, 45))
        self._decoy_freq = float(np.clip(self._decoy_freq, 5, 95))
        self._decoy_strength = float(self.rng.uniform(0.3, 0.6))
        self._decoy_width = float(self.rng.uniform(8, 15))
        self._noise = 0.1

    def _signal(self, freq: float) -> float:
        main = self._main_strength * math.exp(-((freq - self._main_freq) / self._main_width) ** 2)
        decoy = self._decoy_strength * math.exp(-((freq - self._decoy_freq) / self._decoy_width) ** 2)
        return main + decoy

    def get_system_prompt(self) -> str:
        return self._prompts.system

    def get_action_prompt(self) -> str:
        return self._prompts.action.format(
            history=self.format_history(last_n=20),
            round=self.step_count + 1,
        )

    def valid_actions(self) -> list[str]:
        return [str(i) for i in range(101)]

    def parse_action(self, raw: str) -> str | None:
        import re
        nums = re.findall(r'(\d+\.?\d*)', raw)
        if nums:
            val = float(nums[0])
            if 0 <= val <= 100:
                return f"{val:.1f}"
        return None

    def step(self, action: str) -> EnvironmentResult:
        freq = float(action)
        self.step_count += 1

        clean_signal = self._signal(freq)
        noisy_signal = float(np.clip(clean_signal + self.rng.normal(0, self._noise), 0, 1))
        pct = int(round(noisy_signal * 100))
        reward = clean_signal

        flavor = self._prompts.pick_flavor(pct, self.rng)
        summary = f"Dial {freq:.1f} → signal {pct}%"
        self.history.append({"round": self.step_count, "action": action, "reward": reward, "summary": summary})

        feedback = self._prompts.feedback.format(freq=f"{freq:.1f}", signal=pct, flavor=flavor)
        return EnvironmentResult(reward=reward, feedback=feedback)

    def optimal_reward(self) -> float:
        return self._signal(self._main_freq)
