"""Casino Slot Machines environment - stationary multi-armed bandit."""

from __future__ import annotations

import numpy as np

from ..base import Environment, EnvironmentResult
from ...prompts import load_prompt


class CasinoSlotMachines(Environment):
    """6 unlabeled slot machines — one jackpot, one high-variance trap."""

    name = "casino_slot_machines"
    description = "6 unlabeled slot machines with hidden payout distributions"
    is_stationary = True

    MACHINE_NAMES = [
        "Rusty Red",
        "Lucky Lemon",
        "Midnight Blue",
        "Golden Goose",
        "Purple Haze",
        "Silver Streak",
    ]

    def __init__(self, seed: int = 42):
        super().__init__(seed)
        self._prompts = load_prompt(self.name)
        self._means = self.rng.uniform(0.1, 0.9, size=6)
        best = self.rng.integers(0, 6)
        self._means[best] = 0.85 + self.rng.uniform(0, 0.1)
        trap = (best + 3) % 6
        self._means[trap] = 0.4
        self._stds = np.full(6, 0.15)
        self._stds[trap] = 0.4

    def get_system_prompt(self) -> str:
        prompt = self._prompts.system.format(machine_names=", ".join(self.MACHINE_NAMES))
        return self._append_format_instruction(prompt)

    def get_action_prompt(self) -> str:
        return self._prompts.action.format(
            history=self.format_history(last_n=20),
            round=self.step_count + 1,
        )

    def valid_actions(self) -> list[str]:
        return list(self.MACHINE_NAMES)

    def step(self, action: str) -> EnvironmentResult:
        idx = self.MACHINE_NAMES.index(action)
        reward = float(np.clip(self.rng.normal(self._means[idx], self._stds[idx]), 0, 1))
        self.step_count += 1
        summary = f"Played {action} → ${reward:.2f}"
        self.history.append(
            {
                "round": self.step_count,
                "action": action,
                "reward": reward,
                "summary": summary,
            }
        )
        flavor = self._prompts.pick_flavor(reward, self.rng)
        feedback = self._prompts.feedback.format(action=action, reward=reward)
        if flavor:
            feedback += f" {flavor}"
        return EnvironmentResult(reward=reward, feedback=feedback)

    def optimal_reward(self) -> float:
        return float(np.max(self._means))
