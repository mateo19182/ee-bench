"""Clinical Trial environment - non-stationary bandit with resistance buildup."""

from __future__ import annotations

import numpy as np

from ..base import Environment, EnvironmentResult
from ...prompts import load_prompt


class ClinicalTrial(Environment):
    """4 treatments with resistance buildup — cycling is optimal."""

    name = "clinical_trial"
    description = "4 treatments with resistance buildup — cycling is optimal"
    is_stationary = False

    TREATMENTS = ["Compound-A", "Compound-B", "Compound-C", "Compound-D"]

    def __init__(self, seed: int = 42):
        super().__init__(seed)
        self._prompts = load_prompt(self.name)
        self._base_efficacy = self.rng.uniform(0.5, 0.9, size=4)
        self._resistance = np.zeros(4)
        self._consecutive_use = np.zeros(4, dtype=int)
        self._last_action: int | None = None

    def get_system_prompt(self) -> str:
        prompt = self._prompts.system.format(treatment_names=", ".join(self.TREATMENTS))
        return self._append_format_instruction(prompt)

    def get_action_prompt(self) -> str:
        return self._prompts.action.format(
            history=self.format_history(last_n=20),
            round=self.step_count + 1,
        )

    def valid_actions(self) -> list[str]:
        return list(self.TREATMENTS)

    def step(self, action: str) -> EnvironmentResult:
        idx = self.TREATMENTS.index(action)
        self.step_count += 1

        if self._last_action is not None and self._last_action == idx:
            self._consecutive_use[idx] += 1
        else:
            self._consecutive_use[idx] = 1

        for i in range(4):
            if i != idx:
                self._resistance[i] = max(0, self._resistance[i] - 0.08)

        self._resistance[idx] += 0.05 * self._consecutive_use[idx]
        effective = max(0.05, self._base_efficacy[idx] - self._resistance[idx])
        score = float(np.clip(self.rng.normal(effective * 100, 8), 0, 100))

        self._last_action = idx
        pct = int(round(score))
        flavor = self._prompts.pick_flavor(pct, self.rng)
        summary = f"Prescribed {action} → {pct}% improvement"
        self.history.append(
            {
                "round": self.step_count,
                "action": action,
                "reward": score / 100,
                "summary": summary,
            }
        )
        feedback = self._prompts.feedback.format(action=action, score=pct, flavor=flavor)
        return EnvironmentResult(reward=score / 100, feedback=feedback)

    def optimal_reward(self) -> float:
        return float(np.max(self._base_efficacy))
