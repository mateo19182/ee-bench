"""Search and optimization environments.

These differ from bandits: instead of repeated pulls from distributions,
the LLM navigates a space looking for an optimum with limited probes.
"""

from __future__ import annotations

import math

import numpy as np

from .base import Environment, EnvironmentResult


# -----------------------------------------------------------------------
# 1. Treasure Hunter  (2D grid search with proximity hints)
# -----------------------------------------------------------------------

class TreasureHunter(Environment):
    """
    You're an archaeologist searching a 10x10 grid for a buried artifact.
    Each dig gives a proximity signal (warmer/colder).  The LLM must
    balance broad exploration of the grid vs. narrowing down near hot signals.
    """

    name = "treasure_hunter"
    description = "10x10 grid search — dig for treasure using proximity hints"
    is_stationary = True

    def __init__(self, seed: int = 42):
        super().__init__(seed)
        self._target = (int(self.rng.integers(0, 10)), int(self.rng.integers(0, 10)))
        self._best_distance = float('inf')

    def _distance(self, r: int, c: int) -> float:
        return math.sqrt((r - self._target[0]) ** 2 + (c - self._target[1]) ** 2)

    def _signal_strength(self, r: int, c: int) -> float:
        """0-1 signal, 1.0 = on top of treasure."""
        d = self._distance(r, c)
        max_d = math.sqrt(9**2 + 9**2)  # diagonal of grid
        return max(0, 1.0 - d / max_d)

    def get_system_prompt(self) -> str:
        return """You are an archaeologist searching for a legendary artifact buried somewhere in a 10x10 grid (rows 0-9, columns 0-9).

Each round you choose a cell to dig. Your equipment gives you a signal strength reading (0-100):
- 100 = you're right on top of it
- Higher numbers = warmer (closer)
- Lower numbers = colder (farther away)

You also get a directional hint: whether this dig was WARMER or COLDER than your previous best.

Your goal: find the treasure (or get as close as possible) within the given number of rounds.

Each round, respond with ONLY the coordinates as "row,col" (e.g., "3,7"). Nothing else."""

    def get_action_prompt(self) -> str:
        history = self.format_history(last_n=20)
        return f"""Dig history:
{history}

Dig #{self.step_count + 1}: Which cell do you dig?"""

    def valid_actions(self) -> list[str]:
        return [f"{r},{c}" for r in range(10) for c in range(10)]

    def parse_action(self, raw: str) -> str | None:
        import re
        # find two numbers
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

        # reward: signal strength (higher = closer)
        found = (r, c) == self._target
        if found:
            flavor = "YOU FOUND IT! The artifact gleams in the soil!"
        elif signal_pct > 80:
            flavor = "Very strong signal! You're incredibly close."
        elif signal_pct > 60:
            flavor = "The detector is beeping steadily. Getting warm."
        elif signal_pct > 40:
            flavor = "A faint pulse. Something is out there."
        else:
            flavor = "Barely any signal. Cold ground."

        summary = f"Dug ({r},{c}) → signal {signal_pct}% [{direction}]"
        self.history.append({"round": self.step_count, "action": action, "reward": signal, "summary": summary})

        return EnvironmentResult(
            reward=signal,
            feedback=f"Dig at ({r},{c}): Signal strength {signal_pct}%. {direction} than your best. {flavor}",
            info={"found": found},
        )

    def optimal_reward(self) -> float:
        return 1.0  # finding the treasure


# -----------------------------------------------------------------------
# 2. Alchemy Lab  (combinatorial search — mix ingredients)
# -----------------------------------------------------------------------

class AlchemyLab(Environment):
    """
    You're an alchemist mixing 3 ingredients (each 0-9 units) to brew a potion.
    There's a secret optimal recipe. Each attempt gives a potency score.
    The search space is 10^3 = 1000 combinations — too many to brute force.
    The response surface is smooth, so gradient-like strategies work.
    """

    name = "alchemy_lab"
    description = "Mix 3 ingredients to find the optimal potion recipe"
    is_stationary = True

    INGREDIENTS = ["Dragon's Breath", "Moon Dust", "Serpent Oil"]

    def __init__(self, seed: int = 42):
        super().__init__(seed)
        # secret optimal recipe
        self._optimal = self.rng.integers(1, 9, size=3).astype(float)
        # response surface: Gaussian centered on optimal
        self._width = self.rng.uniform(2.0, 4.0, size=3)

    def _potency(self, amounts: tuple[int, int, int]) -> float:
        """Smooth response surface — Gaussian in 3D."""
        diff = np.array(amounts, dtype=float) - self._optimal
        exponent = -np.sum((diff / self._width) ** 2)
        return float(np.exp(exponent))

    def get_system_prompt(self) -> str:
        return f"""You are an alchemist trying to brew the perfect potion. You have three ingredients:
  1. {self.INGREDIENTS[0]}
  2. {self.INGREDIENTS[1]}
  3. {self.INGREDIENTS[2]}

Each ingredient can be added in amounts from 0 to 9 units. Each attempt, you choose amounts for all three and observe the potion's potency (0-100%).

The response is smooth — small changes in amounts cause small changes in potency. But the optimal recipe is unknown.

Your goal: find the recipe with the highest potency using as few attempts as possible.

Each round, respond with ONLY three numbers separated by commas (e.g., "3,5,7"). Nothing else."""

    def get_action_prompt(self) -> str:
        history = self.format_history(last_n=20)
        return f"""Experiment log:
{history}

Attempt #{self.step_count + 1}: What amounts do you try?"""

    def valid_actions(self) -> list[str]:
        # too many to enumerate — we rely on parse_action
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

        if pct > 90:
            flavor = "The potion glows brilliantly! Nearly perfect!"
        elif pct > 70:
            flavor = "A promising brew. The liquid shimmers."
        elif pct > 40:
            flavor = "Mediocre result. The potion is cloudy."
        elif pct > 20:
            flavor = "Weak. The mixture barely reacts."
        else:
            flavor = "Nothing happens. The ingredients just sit there."

        amounts_str = f"{self.INGREDIENTS[0]}={amounts[0]}, {self.INGREDIENTS[1]}={amounts[1]}, {self.INGREDIENTS[2]}={amounts[2]}"
        summary = f"Mixed ({action}) → potency {pct}%"
        self.history.append({"round": self.step_count, "action": action, "reward": reward, "summary": summary})

        return EnvironmentResult(
            reward=reward,
            feedback=f"Recipe: {amounts_str}. Potency: {pct}%. {flavor}",
        )

    def optimal_reward(self) -> float:
        return 1.0


# -----------------------------------------------------------------------
# 3. Radio Tuner  (1D search with noise)
# -----------------------------------------------------------------------

class RadioTuner(Environment):
    """
    You're tuning an old radio dial (0-100) trying to find a hidden
    station.  Signal strength increases as you get closer.  But there's
    static — readings are noisy.  And there might be a secondary weaker
    station that acts as a local optimum / distraction.
    """

    name = "radio_tuner"
    description = "Tune a radio dial (0-100) to find the hidden station"
    is_stationary = True

    def __init__(self, seed: int = 42):
        super().__init__(seed)
        # main station
        self._main_freq = float(self.rng.uniform(15, 85))
        self._main_strength = 1.0
        self._main_width = float(self.rng.uniform(5, 12))
        # decoy station (weaker, might mislead)
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
        return """You are tuning an old analog radio. The dial goes from 0 to 100. Somewhere on the dial is a station broadcasting a signal.

Each round you set the dial to a position and read the signal strength (0-100%). There's static, so readings fluctuate a bit.

WARNING: There might be more than one signal source. A weaker station could mislead you if you settle on it too early.

Your goal: lock onto the strongest possible signal.

Each round, respond with ONLY a number between 0 and 100 (can be a decimal, e.g., "42.5"). Nothing else."""

    def get_action_prompt(self) -> str:
        history = self.format_history(last_n=20)
        return f"""Tuning log:
{history}

Tune #{self.step_count + 1}: Where do you set the dial?"""

    def valid_actions(self) -> list[str]:
        # continuous — rely on parse_action
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
        reward = clean_signal  # use clean for reward/regret

        if pct > 85:
            flavor = "Crystal clear! You can hear every note."
        elif pct > 60:
            flavor = "Decent reception. Music comes through with some crackle."
        elif pct > 30:
            flavor = "Faint. You can tell something's there but can't make it out."
        else:
            flavor = "Just static and white noise."

        summary = f"Dial {freq:.1f} → signal {pct}%"
        self.history.append({"round": self.step_count, "action": action, "reward": reward, "summary": summary})

        return EnvironmentResult(
            reward=reward,
            feedback=f"Dial position: {freq:.1f}. Signal strength: {pct}%. {flavor}",
        )

    def optimal_reward(self) -> float:
        return self._signal(self._main_freq)
