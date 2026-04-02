"""Bandit environments with creative scenarios.

Each environment wraps a multi-armed bandit in a vivid narrative so
the LLM engages with the *scenario* rather than recognising the
underlying math problem and pattern-matching a textbook solution.
"""

from __future__ import annotations

import numpy as np

from .base import Environment, EnvironmentResult
from ..prompts import load_prompt


# -----------------------------------------------------------------------
# 1. Casino Slot Machines  (stationary, classic)
# -----------------------------------------------------------------------

class CasinoSlotMachines(Environment):
    """6 unlabeled slot machines — one jackpot, one high-variance trap."""

    name = "casino_slot_machines"
    description = "6 unlabeled slot machines with hidden payout distributions"
    is_stationary = True

    MACHINE_NAMES = [
        "Rusty Red", "Lucky Lemon", "Midnight Blue",
        "Golden Goose", "Purple Haze", "Silver Streak",
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
        return self._prompts.system.format(machine_names=", ".join(self.MACHINE_NAMES))

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
        self.history.append({"round": self.step_count, "action": action, "reward": reward, "summary": summary})
        flavor = self._prompts.pick_flavor(reward, self.rng)
        feedback = self._prompts.feedback.format(action=action, reward=reward)
        if flavor:
            feedback += f" {flavor}"
        return EnvironmentResult(reward=reward, feedback=feedback)

    def optimal_reward(self) -> float:
        return float(np.max(self._means))


# -----------------------------------------------------------------------
# 2. Restaurant Picker  (stationary, contextual flavor)
# -----------------------------------------------------------------------

class RestaurantPicker(Environment):
    """5 restaurants — tourist trap with flashy name, hidden gem elsewhere."""

    name = "restaurant_picker"
    description = "5 restaurants in a new city — find your favorite"
    is_stationary = True

    RESTAURANTS = [
        "Mama Rosa's Trattoria",
        "The Gilded Fork",
        "Szechuan Alley",
        "Neptune's Catch",
        "Burger Philosophy",
    ]

    def __init__(self, seed: int = 42):
        super().__init__(seed)
        self._prompts = load_prompt(self.name)
        self._means = self.rng.uniform(3.0, 8.5, size=5)
        gem = self.rng.integers(0, 5)
        self._means[gem] = 8.5 + self.rng.uniform(0, 1.0)
        self._means[1] = 3.5  # tourist trap
        self._stds = self.rng.uniform(0.5, 1.5, size=5)

    def get_system_prompt(self) -> str:
        restaurant_list = "\n".join(f"  - {r}" for r in self.RESTAURANTS)
        return self._prompts.system.format(restaurant_list=restaurant_list)

    def get_action_prompt(self) -> str:
        return self._prompts.action.format(
            history=self.format_history(last_n=20),
            round=self.step_count + 1,
        )

    def valid_actions(self) -> list[str]:
        return list(self.RESTAURANTS)

    def step(self, action: str) -> EnvironmentResult:
        idx = self.RESTAURANTS.index(action)
        score = float(np.clip(self.rng.normal(self._means[idx], self._stds[idx]), 1, 10))
        score_int = int(round(score))
        self.step_count += 1

        flavor = self._prompts.pick_flavor(score_int, self.rng)
        summary = f"Ate at {action} → {score_int}/10"
        self.history.append({"round": self.step_count, "action": action, "reward": score / 10, "summary": summary})
        feedback = self._prompts.feedback.format(action=action, score=score_int, flavor=flavor)
        return EnvironmentResult(reward=score / 10, feedback=feedback)

    def optimal_reward(self) -> float:
        return float(np.max(self._means)) / 10


# -----------------------------------------------------------------------
# 3. Clinical Trial  (NON-STATIONARY — drug resistance)
# -----------------------------------------------------------------------

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
        return self._prompts.system.format(treatment_names=", ".join(self.TREATMENTS))

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
        self.history.append({"round": self.step_count, "action": action, "reward": score / 100, "summary": summary})
        feedback = self._prompts.feedback.format(action=action, score=pct, flavor=flavor)
        return EnvironmentResult(reward=score / 100, feedback=feedback)

    def optimal_reward(self) -> float:
        return float(np.max(self._base_efficacy))


# -----------------------------------------------------------------------
# 4. Venture Capitalist  (NON-STATIONARY — market trends)
# -----------------------------------------------------------------------

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
        return self._prompts.system.format(sector_names=", ".join(self.SECTORS))

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
        self.history.append({"round": self.step_count, "action": action, "reward": reward, "summary": summary})
        feedback = self._prompts.feedback.format(action=action, **{"return": f"{ret:+.1f}"}, flavor=flavor)
        return EnvironmentResult(reward=reward, feedback=feedback)

    def optimal_reward(self) -> float:
        return float(np.max(self._hotness))


# -----------------------------------------------------------------------
# 5. Ocean Fishing  (stationary, but with spatial correlation)
# -----------------------------------------------------------------------

class OceanFishing(Environment):
    """8 fishing spots with spatial correlation — nearby spots have similar yields."""

    name = "ocean_fishing"
    description = "8 fishing spots with spatial correlation — nearby spots have similar yields"
    is_stationary = True

    SPOTS = [
        "Coral Cove", "Driftwood Bay", "Eagle Point", "Foghorn Reef",
        "Gull Island", "Hermit's Inlet", "Ironside Shelf", "Jade Lagoon",
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
        spot_list = "\n".join(f"  {i+1}. {s}" for i, s in enumerate(self.SPOTS))
        return self._prompts.system.format(spot_list=spot_list)

    def get_action_prompt(self) -> str:
        return self._prompts.action.format(
            history=self.format_history(last_n=20),
            round=self.step_count + 1,
        )

    def valid_actions(self) -> list[str]:
        return list(self.SPOTS)

    def step(self, action: str) -> EnvironmentResult:
        idx = self.SPOTS.index(action)
        catch = float(np.clip(self.rng.normal(self._means[idx] * 100, self._stds[idx] * 100), 0, 100))
        catch_int = int(round(catch))
        self.step_count += 1
        reward = catch / 100

        flavor = self._prompts.pick_flavor(catch_int, self.rng)
        summary = f"Fished at {action} → {catch_int} fish"
        self.history.append({"round": self.step_count, "action": action, "reward": reward, "summary": summary})
        feedback = self._prompts.feedback.format(action=action, catch=catch_int, flavor=flavor)
        return EnvironmentResult(reward=reward, feedback=feedback)

    def optimal_reward(self) -> float:
        return float(np.max(self._means))
