from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class EnvironmentResult:
    """What the environment returns after an action."""
    reward: float
    feedback: str  # natural-language feedback shown to the LLM
    info: dict[str, Any] = field(default_factory=dict)


class Environment(abc.ABC):
    """Base class for all explore/exploit environments."""

    name: str  # short id
    description: str  # one-liner
    is_stationary: bool = True

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.step_count = 0
        self.history: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def get_system_prompt(self) -> str:
        """The scenario framing + rules shown once at the start."""

    @abc.abstractmethod
    def get_action_prompt(self) -> str:
        """Per-round prompt asking the LLM to choose an action.
        Should describe available actions clearly."""

    @abc.abstractmethod
    def valid_actions(self) -> list[str]:
        """List of action labels the LLM can choose from."""

    @abc.abstractmethod
    def step(self, action: str) -> EnvironmentResult:
        """Execute the action, return result."""

    @abc.abstractmethod
    def optimal_reward(self) -> float:
        """The best expected reward for the current step.
        Used to compute regret."""

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def parse_action(self, raw: str) -> str | None:
        """Try to extract a valid action from LLM output.
        Returns None if no valid action found."""
        raw_lower = raw.strip().lower()
        actions = self.valid_actions()

        # exact match
        for a in actions:
            if a.lower() == raw_lower:
                return a

        # check if any action label appears in the response
        found = []
        for a in actions:
            if a.lower() in raw_lower:
                found.append(a)

        if len(found) == 1:
            return found[0]

        # try to find a number if actions are numbered
        import re
        numbers = re.findall(r'\b(\d+)\b', raw)
        for num_str in numbers:
            idx = int(num_str) - 1  # 1-indexed
            if 0 <= idx < len(actions):
                return actions[idx]

        return None

    def format_history(self, last_n: int | None = None) -> str:
        """Format the action/reward history as text."""
        entries = self.history if last_n is None else self.history[-last_n:]
        if not entries:
            return "No actions taken yet."
        lines = []
        for i, h in enumerate(entries, 1):
            lines.append(f"Round {h['round']}: {h['summary']}")
        return "\n".join(lines)
