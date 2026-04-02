"""Environment implementations for explore/exploit benchmarks."""

from .base import Environment, EnvironmentResult
from .bandits import (
    CasinoSlotMachines,
    ClinicalTrial,
    OceanFishing,
    RestaurantPicker,
    VentureCapitalist,
)
from .search import AlchemyLab, RadioTuner, TreasureHunter

ALL_ENVIRONMENTS: list[type[Environment]] = [
    # Bandits - stationary
    CasinoSlotMachines,
    RestaurantPicker,
    OceanFishing,
    # Bandits - non-stationary
    ClinicalTrial,
    VentureCapitalist,
    # Search/optimization
    TreasureHunter,
    AlchemyLab,
    RadioTuner,
]

__all__ = [
    "Environment",
    "EnvironmentResult",
    "ALL_ENVIRONMENTS",
    "CasinoSlotMachines",
    "RestaurantPicker",
    "OceanFishing",
    "ClinicalTrial",
    "VentureCapitalist",
    "TreasureHunter",
    "AlchemyLab",
    "RadioTuner",
]
