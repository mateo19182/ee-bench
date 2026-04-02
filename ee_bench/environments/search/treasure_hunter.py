"""Treasure Hunter environment - 2D grid search with proximity hints."""

from __future__ import annotations

import math
import re

import numpy as np

from ..base import Environment, EnvironmentResult
from ...prompts import load_prompt


class TreasureHunter(Environment):
    """10x10 grid search — dig for treasure using proximity hints."""

    name = "treasure_hunter"
    description = "10x10 grid search — dig for treasure using proximity hints"
    is_stationary = True

    def __init__(self, seed: int = 42):
        super().__init__(seed)
        self._prompts = load_prompt(self.name)
        self._target = (int(self.rng.integers(0, 10)), int(self.rng.integers(0, 10)))
        self._best_distance = float("inf")

    def _distance(self, r: int, c: int) -> float:
        return math.sqrt((r - self._target[0]) ** 2 + (c - self._target[1]) ** 2)

    def _signal_strength(self, r: int, c: int) -> float:
        d = self._distance(r, c)
        max_d = math.sqrt(9**2 + 9**2)
        return max(0, 1.0 - d / max_d)

    def get_system_prompt(self) -> str:
        prompt = self._prompts.system
        return self._append_format_instruction(prompt)

    def get_action_prompt(self) -> str:
        return self._prompts.action.format(
            history=self.format_history(last_n=20),
            round=self.step_count + 1,
        )

    def valid_actions(self) -> list[str]:
        return [f"{r},{c}" for r in range(10) for c in range(10)]

    def parse_action(self, raw: str) -> str | None:
        nums = re.findall(r"(\d+)", raw)
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
        self.history.append(
            {
                "round": self.step_count,
                "action": action,
                "reward": signal,
                "summary": summary,
            }
        )

        feedback = self._prompts.feedback.format(
            row=r,
            col=c,
            signal=signal_pct,
            direction=direction,
            flavor=flavor,
        )
        return EnvironmentResult(reward=signal, feedback=feedback, info={"found": found})

    def optimal_reward(self) -> float:
        return 1.0
