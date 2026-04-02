"""Bandit environments for explore/exploit scenarios."""

from .casino_slot_machines import CasinoSlotMachines
from .clinical_trial import ClinicalTrial
from .ocean_fishing import OceanFishing
from .restaurant_picker import RestaurantPicker
from .venture_capitalist import VentureCapitalist

__all__ = [
    "CasinoSlotMachines",
    "ClinicalTrial",
    "OceanFishing",
    "RestaurantPicker",
    "VentureCapitalist",
]
