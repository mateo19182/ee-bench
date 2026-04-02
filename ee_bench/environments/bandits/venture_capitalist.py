"""Venture Capitalist environment - non-stationary bandit with shifting market trends."""

from __future__ import annotations

import numpy as np

from ..base import Environment, EnvironmentResult
from ...prompts import load_prompt


class VentureCapitalist(Environment):
    """Invest across 5 sectors with shifting market trends."""

    name = "venture_capitalist"
    description = "Invest across 5 sectors with shifting market trends"
    is_stationary = False

    SECTORS = ["AI/ML", "Biotech", "Climate Tech", "Fintech", "Space"]

    def __init__(self, seed: int = 42):
        super().__init__(seed)
        self._prompts = load_prompt(self.name)
        self._hotness = self.rng.uniform(0.3, 0.7, size=5)
        self._drift_speed = 0.04
        self._hot_idx = self.rng.integers(0, 5)
        self._hotness[self._hot_idx] = 0.85
        self._regime_change_at = self.rng.integers(30, 80)
        self._regime_changed = False

    def _tick_market(self):
        self._hotness += self.rng.normal(0, self._drift_speed, size=5)
        if self.step_count == self._regime_change_at and not self._regime_changed:
            self._hotness[self._hot_idx] -= 0.3
            new_hot = (self._hot_idx + self.rng.integers(1, 5)) % 5
            self._hotness[new_hot] += 0.3
            self._hot_idx = new_hot
            self._regime_changed = True
        self._hotness = np.clip(self._hotness, 0.05, 0.95)

    def get_system_prompt(self) -> str:
        prompt = self._prompts.system.format(sector_names=", ".join(self.SECTORS))
        return self._append_format_instruction(prompt)

    def get_action_prompt(self) -> str:
        return self._prompts.action.format(
            history=self.format_history(last_n=20),
            round=self.step_count + 1,
        )

    def valid_actions(self) -> list[str]:
        return list(self.SECTORS)

    def step(self, action: str) -> EnvironmentResult:
        idx = self.SECTORS.index(action)
        self.step_count += 1
        self._tick_market()

        ret = float(self.rng.normal(self._hotness[idx] * 30, 5))
        ret = round(ret, 1)
        reward = max(0, ret) / 30

        flavor = self._prompts.pick_flavor(ret, self.rng)
        summary = f"Invested in {action} → {ret:+.1f}% return"
        self.history.append(
            {
                "round": self.step_count,
                "action": action,
                "reward": reward,
                "summary": summary,
            }
        )
        feedback = self._prompts.feedback.format(
            action=action, **{"return": f"{ret:+.1f}"}, flavor=flavor
        )
        return EnvironmentResult(reward=reward, feedback=feedback)

    def optimal_reward(self) -> float:
        return float(np.max(self._hotness))
