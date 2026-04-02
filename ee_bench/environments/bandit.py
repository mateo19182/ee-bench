"""Bandit environments with creative scenarios.

Each environment wraps a multi-armed bandit in a vivid narrative so
the LLM engages with the *scenario* rather than recognising the
underlying math problem and pattern-matching a textbook solution.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .base import Environment, EnvironmentResult


# -----------------------------------------------------------------------
# 1. Casino Slot Machines  (stationary, classic)
# -----------------------------------------------------------------------

class CasinoSlotMachines(Environment):
    """
    You're in a quirky underground casino with 6 unlabeled slot machines.
    Each machine has a different (hidden) payout distribution.
    Some are duds, one is the jackpot machine — but which one?
    """

    name = "casino_slot_machines"
    description = "6 unlabeled slot machines with hidden payout distributions"
    is_stationary = True

    MACHINE_NAMES = [
        "Rusty Red", "Lucky Lemon", "Midnight Blue",
        "Golden Goose", "Purple Haze", "Silver Streak",
    ]

    def __init__(self, seed: int = 42):
        super().__init__(seed)
        # hidden means and std devs — one clearly best, some traps
        self._means = self.rng.uniform(0.1, 0.9, size=6)
        # make one arm clearly dominant
        best = self.rng.integers(0, 6)
        self._means[best] = 0.85 + self.rng.uniform(0, 0.1)
        # make one arm a "trap" — high variance, mediocre mean
        trap = (best + 3) % 6
        self._means[trap] = 0.4
        self._stds = np.full(6, 0.15)
        self._stds[trap] = 0.4  # high variance trap

    def get_system_prompt(self) -> str:
        names = ", ".join(self.MACHINE_NAMES)
        return f"""You are in an underground casino with 6 slot machines: {names}.

Each machine has a different hidden payout profile. Some pay well consistently, others are unreliable, and some are just bad. You have no idea which is which — you have to figure it out by playing.

Each round you pull one machine's lever and receive a payout between $0.00 and $1.00.

Your goal: maximize your total earnings over all rounds.

IMPORTANT: Each round, respond with ONLY the name of the machine you want to play. Nothing else."""

    def get_action_prompt(self) -> str:
        history = self.format_history(last_n=20)
        return f"""History of your last plays:
{history}

Round {self.step_count + 1}: Which machine do you play?"""

    def valid_actions(self) -> list[str]:
        return list(self.MACHINE_NAMES)

    def step(self, action: str) -> EnvironmentResult:
        idx = self.MACHINE_NAMES.index(action)
        reward = float(np.clip(self.rng.normal(self._means[idx], self._stds[idx]), 0, 1))
        self.step_count += 1
        summary = f"Played {action} → ${reward:.2f}"
        self.history.append({"round": self.step_count, "action": action, "reward": reward, "summary": summary})
        return EnvironmentResult(
            reward=reward,
            feedback=f"You pulled {action} and won ${reward:.2f}.",
        )

    def optimal_reward(self) -> float:
        return float(np.max(self._means))


# -----------------------------------------------------------------------
# 2. Restaurant Picker  (stationary, contextual flavor)
# -----------------------------------------------------------------------

class RestaurantPicker(Environment):
    """
    You just moved to a new city. Every night you eat out at one of 5
    restaurants.  You rate your experience 1-10.  Some places are
    hidden gems, others are tourist traps with flashy names.
    """

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

    # flavor text for each outcome range
    REVIEWS = {
        (8, 10): [
            "Absolutely divine. You lick the plate clean.",
            "The chef outdid themselves. A spiritual experience.",
            "You consider writing a love letter to the kitchen.",
        ],
        (5, 7): [
            "Solid meal. Nothing to complain about.",
            "Decent. You'd come back if nothing else was open.",
            "Fine. The ambiance carried it.",
        ],
        (1, 4): [
            "Disappointing. You regret not cooking at home.",
            "The menu promised more than the kitchen delivered.",
            "You leave hungry and slightly offended.",
        ],
    }

    def __init__(self, seed: int = 42):
        super().__init__(seed)
        self._means = self.rng.uniform(3.0, 8.5, size=5)
        # one hidden gem
        gem = self.rng.integers(0, 5)
        self._means[gem] = 8.5 + self.rng.uniform(0, 1.0)
        # one tourist trap — flashy name (The Gilded Fork) but meh food
        self._means[1] = 3.5
        self._stds = self.rng.uniform(0.5, 1.5, size=5)

    def get_system_prompt(self) -> str:
        names = "\n".join(f"  - {r}" for r in self.RESTAURANTS)
        return f"""You just moved to a new city and don't know any of the restaurants. Every evening you pick one place to eat dinner. After each meal you rate your experience from 1 to 10.

Available restaurants:
{names}

Some places look fancy but disappoint. Others look unassuming but are incredible. The only way to know is to try them.

Your goal: have the best dining experiences possible over all your evenings out.

Each round, respond with ONLY the restaurant name. Nothing else."""

    def get_action_prompt(self) -> str:
        history = self.format_history(last_n=20)
        return f"""Your dining history:
{history}

Evening {self.step_count + 1}: Where do you eat tonight?"""

    def valid_actions(self) -> list[str]:
        return list(self.RESTAURANTS)

    def step(self, action: str) -> EnvironmentResult:
        idx = self.RESTAURANTS.index(action)
        score = float(np.clip(self.rng.normal(self._means[idx], self._stds[idx]), 1, 10))
        score_int = int(round(score))
        self.step_count += 1

        # pick flavor text
        for (lo, hi), texts in self.REVIEWS.items():
            if lo <= score_int <= hi:
                flavor = self.rng.choice(texts)
                break
        else:
            flavor = "An unremarkable evening."

        summary = f"Ate at {action} → {score_int}/10"
        self.history.append({"round": self.step_count, "action": action, "reward": score / 10, "summary": summary})
        return EnvironmentResult(
            reward=score / 10,
            feedback=f"You dined at {action}. Your rating: {score_int}/10. {flavor}",
        )

    def optimal_reward(self) -> float:
        return float(np.max(self._means)) / 10


# -----------------------------------------------------------------------
# 3. Clinical Trial  (NON-STATIONARY — drug resistance)
# -----------------------------------------------------------------------

class ClinicalTrial(Environment):
    """
    You're a doctor trying 4 experimental treatments on patients with
    a chronic condition.  Treatments work at first, but patients can
    develop resistance — effectiveness degrades with consecutive use.
    The twist: switching away and coming back partially resets resistance.
    Optimal play requires *cycling* treatments, not just finding "the best one".
    """

    name = "clinical_trial"
    description = "4 treatments with resistance buildup — cycling is optimal"
    is_stationary = False

    TREATMENTS = ["Compound-A", "Compound-B", "Compound-C", "Compound-D"]

    def __init__(self, seed: int = 42):
        super().__init__(seed)
        self._base_efficacy = self.rng.uniform(0.5, 0.9, size=4)
        # resistance: how much consecutive use degrades efficacy
        self._resistance = np.zeros(4)
        self._consecutive_use = np.zeros(4, dtype=int)
        self._last_action: int | None = None

    def get_system_prompt(self) -> str:
        names = ", ".join(self.TREATMENTS)
        return f"""You are a physician managing a chronic condition patient. You have 4 experimental treatments available: {names}.

Each week you prescribe one treatment and observe how much the patient improves (0-100% improvement score).

IMPORTANT: Treatments may lose effectiveness over time if used repeatedly. The body can develop resistance. However, if you stop using a treatment for a while, resistance may partially fade.

Your goal: maximize the patient's total improvement across all weeks.

Each round, respond with ONLY the treatment name. Nothing else."""

    def get_action_prompt(self) -> str:
        history = self.format_history(last_n=20)
        return f"""Treatment history:
{history}

Week {self.step_count + 1}: Which treatment do you prescribe?"""

    def valid_actions(self) -> list[str]:
        return list(self.TREATMENTS)

    def step(self, action: str) -> EnvironmentResult:
        idx = self.TREATMENTS.index(action)
        self.step_count += 1

        # update resistance
        if self._last_action is not None and self._last_action == idx:
            self._consecutive_use[idx] += 1
        else:
            self._consecutive_use[idx] = 1

        # resistance recovery for unused treatments
        for i in range(4):
            if i != idx:
                self._resistance[i] = max(0, self._resistance[i] - 0.08)

        # resistance buildup
        self._resistance[idx] += 0.05 * self._consecutive_use[idx]

        effective = max(0.05, self._base_efficacy[idx] - self._resistance[idx])
        score = float(np.clip(self.rng.normal(effective * 100, 8), 0, 100))

        self._last_action = idx
        pct = int(round(score))
        summary = f"Prescribed {action} → {pct}% improvement"
        self.history.append({"round": self.step_count, "action": action, "reward": score / 100, "summary": summary})

        if pct > 70:
            flavor = "The patient responds well. Visible improvement."
        elif pct > 40:
            flavor = "Moderate response. Some improvement noted."
        else:
            flavor = "Poor response. The patient shows minimal improvement."

        return EnvironmentResult(
            reward=score / 100,
            feedback=f"Treatment: {action}. Improvement score: {pct}%. {flavor}",
        )

    def optimal_reward(self) -> float:
        return float(np.max(self._base_efficacy))


# -----------------------------------------------------------------------
# 4. Venture Capitalist  (NON-STATIONARY — market trends)
# -----------------------------------------------------------------------

class VentureCapitalist(Environment):
    """
    You're a VC deciding which startup sector to invest in each quarter.
    Sectors go through boom/bust cycles — what's hot now won't be hot
    forever.  The LLM must detect trend shifts and reallocate.
    """

    name = "venture_capitalist"
    description = "Invest across 5 sectors with shifting market trends"
    is_stationary = False

    SECTORS = ["AI/ML", "Biotech", "Climate Tech", "Fintech", "Space"]

    def __init__(self, seed: int = 42):
        super().__init__(seed)
        # each sector has a "hotness" that drifts via random walk
        self._hotness = self.rng.uniform(0.3, 0.7, size=5)
        self._drift_speed = 0.04
        # one sector starts as the clear winner
        self._hot_idx = self.rng.integers(0, 5)
        self._hotness[self._hot_idx] = 0.85
        # schedule a regime change at a random step
        self._regime_change_at = self.rng.integers(30, 80)
        self._regime_changed = False

    def _tick_market(self):
        """Advance market dynamics."""
        # random walk drift
        self._hotness += self.rng.normal(0, self._drift_speed, size=5)

        # regime change: the hot sector crashes, a random other one booms
        if self.step_count == self._regime_change_at and not self._regime_changed:
            self._hotness[self._hot_idx] -= 0.3
            new_hot = (self._hot_idx + self.rng.integers(1, 5)) % 5
            self._hotness[new_hot] += 0.3
            self._hot_idx = new_hot
            self._regime_changed = True

        self._hotness = np.clip(self._hotness, 0.05, 0.95)

    def get_system_prompt(self) -> str:
        names = ", ".join(self.SECTORS)
        return f"""You are a venture capitalist. Each quarter you choose one sector to focus your investments on: {names}.

After each quarter you see your returns (as a percentage). Sector performance shifts over time — today's winner may be tomorrow's loser. Markets are unpredictable, but trends exist if you pay attention.

Your goal: maximize total returns across all quarters.

Each round, respond with ONLY the sector name. Nothing else."""

    def get_action_prompt(self) -> str:
        history = self.format_history(last_n=20)
        return f"""Investment history:
{history}

Quarter {self.step_count + 1}: Which sector do you invest in?"""

    def valid_actions(self) -> list[str]:
        return list(self.SECTORS)

    def step(self, action: str) -> EnvironmentResult:
        idx = self.SECTORS.index(action)
        self.step_count += 1
        self._tick_market()

        ret = float(self.rng.normal(self._hotness[idx] * 30, 5))  # % return
        ret = round(ret, 1)
        reward = max(0, ret) / 30  # normalized

        summary = f"Invested in {action} → {ret:+.1f}% return"
        self.history.append({"round": self.step_count, "action": action, "reward": reward, "summary": summary})

        if ret > 20:
            flavor = "Excellent quarter. Your LPs are thrilled."
        elif ret > 10:
            flavor = "Solid returns. The fund is performing well."
        elif ret > 0:
            flavor = "Modest gains. Could be worse."
        else:
            flavor = "Ouch. The sector underperformed. Your LPs are calling."

        return EnvironmentResult(
            reward=reward,
            feedback=f"Sector: {action}. Return: {ret:+.1f}%. {flavor}",
        )

    def optimal_reward(self) -> float:
        return float(np.max(self._hotness))


# -----------------------------------------------------------------------
# 5. Ocean Fishing  (stationary, but with spatial correlation)
# -----------------------------------------------------------------------

class OceanFishing(Environment):
    """
    You're a fisher choosing which of 8 fishing spots to visit each day.
    Spots that are close together tend to have similar fish populations.
    This means trying one spot gives you *partial* info about neighbors.
    Tests whether the LLM can do spatial reasoning in exploration.
    """

    name = "ocean_fishing"
    description = "8 fishing spots with spatial correlation — nearby spots have similar yields"
    is_stationary = True

    SPOTS = [
        "Coral Cove", "Driftwood Bay", "Eagle Point", "Foghorn Reef",
        "Gull Island", "Hermit's Inlet", "Ironside Shelf", "Jade Lagoon",
    ]

    def __init__(self, seed: int = 42):
        super().__init__(seed)
        # spots are arranged in a ring — adjacent spots are correlated
        base = self.rng.uniform(0.2, 0.5, size=8)
        # create a "hotspot" — a peak that spreads to neighbors
        peak = self.rng.integers(0, 8)
        for offset in range(-2, 3):
            idx = (peak + offset) % 8
            base[idx] += max(0, 0.4 - 0.15 * abs(offset))
        self._means = np.clip(base, 0.1, 0.95)
        self._stds = np.full(8, 0.12)

    def get_system_prompt(self) -> str:
        spot_list = "\n".join(f"  {i+1}. {s}" for i, s in enumerate(self.SPOTS))
        return f"""You are a fisher working the coast. There are 8 fishing spots, listed in order along the coastline:

{spot_list}

Each day you sail to one spot and haul in your catch (scored 0-100 fish).

HINT: Fish populations tend to be similar at nearby spots along the coast. If one spot is good, its neighbors might be good too. But the best spot might be anywhere.

Your goal: catch as many fish as possible across all days.

Each round, respond with ONLY the spot name. Nothing else."""

    def get_action_prompt(self) -> str:
        history = self.format_history(last_n=20)
        return f"""Fishing log:
{history}

Day {self.step_count + 1}: Where do you fish today?"""

    def valid_actions(self) -> list[str]:
        return list(self.SPOTS)

    def step(self, action: str) -> EnvironmentResult:
        idx = self.SPOTS.index(action)
        catch = float(np.clip(self.rng.normal(self._means[idx] * 100, self._stds[idx] * 100), 0, 100))
        catch_int = int(round(catch))
        self.step_count += 1
        reward = catch / 100

        summary = f"Fished at {action} → {catch_int} fish"
        self.history.append({"round": self.step_count, "action": action, "reward": reward, "summary": summary})

        if catch_int > 75:
            flavor = "The nets are bursting! Incredible haul."
        elif catch_int > 50:
            flavor = "A good day on the water."
        elif catch_int > 25:
            flavor = "Slim pickings today."
        else:
            flavor = "Almost nothing. The sea was ungenerous."

        return EnvironmentResult(
            reward=reward,
            feedback=f"Spot: {action}. Catch: {catch_int} fish. {flavor}",
        )

    def optimal_reward(self) -> float:
        return float(np.max(self._means))
