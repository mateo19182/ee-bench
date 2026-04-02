"""Ocean Fishing environment - stationary bandit with spatial correlation."""

from __future__ import annotations

import numpy as np

from ..base import Environment, EnvironmentResult
from ...prompts import load_prompt


class OceanFishing(Environment):
    """8 fishing spots with spatial correlation — nearby spots have similar yields."""

    name = "ocean_fishing"
    description = "8 fishing spots with spatial correlation — nearby spots have similar yields"
    is_stationary = True

    SPOTS = [
        "Coral Cove",
        "Driftwood Bay",
        "Eagle Point",
        "Foghorn Reef",
        "Gull Island",
        "Hermit's Inlet",
        "Ironside Shelf",
        "Jade Lagoon",
    ]

    def __init__(self, seed: int = 42):
        super().__init__(seed)
        self._prompts = load_prompt(self.name)
        base = self.rng.uniform(0.2, 0.5, size=8)
        peak = self.rng.integers(0, 8)
        for offset in range(-2, 3):
            idx = (peak + offset) % 8
            base[idx] += max(0, 0.4 - 0.15 * abs(offset))
        self._means = np.clip(base, 0.1, 0.95)
        self._stds = np.full(8, 0.12)

    def get_system_prompt(self) -> str:
        spot_list = "\n".join(f"  {i + 1}. {s}" for i, s in enumerate(self.SPOTS))
        prompt = self._prompts.system.format(spot_list=spot_list)
        return self._append_format_instruction(prompt)

    def get_action_prompt(self) -> str:
        return self._prompts.action.format(
            history=self.format_history(last_n=20),
            round=self.step_count + 1,
        )

    def valid_actions(self) -> list[str]:
        return list(self.SPOTS)

    def step(self, action: str) -> EnvironmentResult:
        idx = self.SPOTS.index(action)
        catch = float(
            np.clip(self.rng.normal(self._means[idx] * 100, self._stds[idx] * 100), 0, 100)
        )
        catch_int = int(round(catch))
        self.step_count += 1
        reward = catch / 100

        flavor = self._prompts.pick_flavor(catch_int, self.rng)
        summary = f"Fished at {action} → {catch_int} fish"
        self.history.append(
            {
                "round": self.step_count,
                "action": action,
                "reward": reward,
                "summary": summary,
            }
        )
        feedback = self._prompts.feedback.format(action=action, catch=catch_int, flavor=flavor)
        return EnvironmentResult(reward=reward, feedback=feedback)

    def optimal_reward(self) -> float:
        return float(np.max(self._means))
