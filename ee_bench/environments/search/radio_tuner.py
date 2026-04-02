"""Radio Tuner environment - 1D search for hidden station with noise."""

from __future__ import annotations

import math
import re

import numpy as np

from ..base import Environment, EnvironmentResult
from ...prompts import load_prompt


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
        self._decoy_freq = float(
            self._main_freq + self.rng.choice([-1, 1]) * self.rng.uniform(25, 45)
        )
        self._decoy_freq = float(np.clip(self._decoy_freq, 5, 95))
        self._decoy_strength = float(self.rng.uniform(0.3, 0.6))
        self._decoy_width = float(self.rng.uniform(8, 15))
        self._noise = 0.1

    def _signal(self, freq: float) -> float:
        main = self._main_strength * math.exp(-(((freq - self._main_freq) / self._main_width) ** 2))
        decoy = self._decoy_strength * math.exp(
            -(((freq - self._decoy_freq) / self._decoy_width) ** 2)
        )
        return main + decoy

    def get_system_prompt(self) -> str:
        prompt = self._prompts.system
        return self._append_format_instruction(prompt)

    def get_action_prompt(self) -> str:
        return self._prompts.action.format(
            history=self.format_history(last_n=20),
            round=self.step_count + 1,
        )

    def valid_actions(self) -> list[str]:
        return [str(i) for i in range(101)]

    def parse_action(self, raw: str) -> str | None:
        nums = re.findall(r"(\d+\.?\d*)", raw)
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
        self.history.append(
            {
                "round": self.step_count,
                "action": action,
                "reward": reward,
                "summary": summary,
            }
        )

        feedback = self._prompts.feedback.format(freq=f"{freq:.1f}", signal=pct, flavor=flavor)
        return EnvironmentResult(reward=reward, feedback=feedback)

    def optimal_reward(self) -> float:
        return self._signal(self._main_freq)
