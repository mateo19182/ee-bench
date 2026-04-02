from __future__ import annotations

import abc
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np


# Regex patterns to match thinking/reasoning blocks in various formats
THINK_PATTERNS = [
    re.compile(r"```think\n.*?```", re.DOTALL | re.IGNORECASE),
    re.compile(r"<thinking>.*?</thinking>", re.DOTALL | re.IGNORECASE),
    re.compile(r"<reasoning>.*?</reasoning>", re.DOTALL | re.IGNORECASE),
    re.compile(r"\[thinking\].*?\[/thinking\]", re.DOTALL | re.IGNORECASE),
]


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
        Returns None if no valid action found.

        Strips thinking/reasoning blocks and prefers the last mention
        of a valid action (since that's likely the final choice after analysis).
        """
        # Strip thinking blocks
        cleaned = raw
        for pattern in THINK_PATTERNS:
            cleaned = pattern.sub("", cleaned)

        cleaned_lower = cleaned.strip().lower()
        actions = self.valid_actions()

        # exact match
        for a in actions:
            if a.lower() == cleaned_lower:
                return a

        # Find all occurrences of action names with their positions
        # Return the LAST one (likely the final choice after analysis)
        found_positions = []
        for a in actions:
            pattern = re.compile(re.escape(a.lower()))
            for match in pattern.finditer(cleaned_lower):
                found_positions.append((match.start(), a))

        if found_positions:
            # Sort by position and return the last one
            found_positions.sort(key=lambda x: x[0])
            return found_positions[-1][1]

        # try to find a number if actions are numbered
        numbers = re.findall(r"\b(\d+)\b", cleaned)
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

    def _append_format_instruction(self, prompt: str) -> str:
        """Append standard format instruction to a prompt."""
        actions = self.valid_actions()
        if len(actions) <= 20:
            instruction = f"\n\nWhen responding, output ONLY the exact name from this list: {', '.join(actions)}"
        else:
            instruction = "\n\nWhen responding, output ONLY the exact name of your choice."
        return prompt + instruction
