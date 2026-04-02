from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment run."""

    # Model
    model: str  # OpenRouter model id, e.g. "anthropic/claude-sonnet-4"
    temperature: float = 0.7

    # Environment
    environment: str = "casino_slot_machines"
    seed: int = 42

    # Horizons to test
    horizons: list[int] = field(default_factory=lambda: [20, 100, 500])

    # Repetitions per (env, horizon, temp) combo for statistical power
    repetitions: int = 5

    # Provider
    api_key: str = ""
    base_url: str = "https://openrouter.ai/api/v1"

    # Output
    results_dir: str = "results"


@dataclass
class SweepConfig:
    """Configuration for a parameter sweep."""

    models: list[str] = field(default_factory=list)
    temperatures: list[float] = field(default_factory=lambda: [0.0, 0.3, 0.7, 1.0])
    environments: list[str] = field(default_factory=list)
    horizons: list[int] = field(default_factory=lambda: [20, 100, 500])
    repetitions: int = 5
    seed: int = 42
    api_key: str = ""
    base_url: str = "https://openrouter.ai/api/v1"
    results_dir: str = "results"
