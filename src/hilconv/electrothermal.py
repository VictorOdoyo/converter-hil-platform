from dataclasses import dataclass

import numpy as np

from .core import finite, positive


@dataclass(frozen=True)
class Device:
    resistance_25_ohm: float = 0.02
    resistance_coefficient_k: float = 0.004
    turnon_reference_j: float = 20e-6
    turnoff_reference_j: float = 15e-6
    reference_current_a: float = 10
    reference_voltage_v: float = 48
    switching_temperature_coefficient_k: float = 0.002

    def __post_init__(self):
        for key in ("resistance_25_ohm", "reference_current_a", "reference_voltage_v"):
            positive(getattr(self, key))
        for key in (
            "resistance_coefficient_k",
            "turnon_reference_j",
            "turnoff_reference_j",
            "switching_temperature_coefficient_k",
        ):
            positive(getattr(self, key), allow_zero=True)


@dataclass(frozen=True)
class Thermal:
    junction_capacity_j_k: float = 0.5
    case_capacity_j_k: float = 10
    junction_case_k_w: float = 0.7
    case_ambient_k_w: float = 3

    def __post_init__(self):
        for key in self.__dataclass_fields__:
            positive(getattr(self, key))

    @property
    def total_resistance(self):
        return self.junction_case_k_w + self.case_ambient_k_w


def resistance(junction_c, device=None):
    p = Device() if device is None else device
    t = finite(junction_c, real=True)
    if np.any(t <= -273.15):
        raise ValueError("physical junction temperature required")
    factor = 1 + p.resistance_coefficient_k * (t - 25)
    if np.any(factor <= 0):
        raise ValueError("temperature lies outside positive resistance model range")
    return p.resistance_25_ohm * factor
