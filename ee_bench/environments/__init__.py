from .base import Environment, EnvironmentResult
from .bandit import (
    CasinoSlotMachines,
    RestaurantPicker,
    ClinicalTrial,
    VentureCapitalist,
    OceanFishing,
)
from .search import (
    TreasureHunter,
    AlchemyLab,
    RadioTuner,
)

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
