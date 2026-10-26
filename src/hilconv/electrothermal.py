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


def conduction_loss(rms_current_a, junction_c, device=None):
    current = finite(rms_current_a, real=True)
    if np.any(current < 0):
        raise ValueError("nonnegative RMS current required")
    return current**2 * resistance(junction_c, device)


def switching_energy(turnon_current_a, turnoff_current_a, bus_v, junction_c, device=None):
    p = Device() if device is None else device
    voltage = positive(bus_v)
    temperature = float(finite(junction_c, real=True))
    if temperature <= -273.15:
        raise ValueError("physical commutation temperature required")
    factor = 1 + p.switching_temperature_coefficient_k * (temperature - 25)
    if factor < 0:
        raise ValueError("temperature outside switching-energy model")
    on = p.turnon_reference_j * abs(float(finite(turnon_current_a, real=True))) / p.reference_current_a
    off = p.turnoff_reference_j * abs(float(finite(turnoff_current_a, real=True))) / p.reference_current_a
    return (on + off) * voltage / p.reference_voltage_v * factor


def thermal_step(temperatures_c, power_w, ambient_c, duration_s, model=None):
    from scipy.linalg import expm

    p = Thermal() if model is None else model
    t = finite(temperatures_c, real=True)
    ambient = float(finite(ambient_c, real=True))
    power = positive(power_w, allow_zero=True)
    dt = positive(duration_s)
    if t.shape != (2,) or min(*t, ambient) <= -273.15:
        raise ValueError("junction case and ambient temperatures required")
    cj, cc, rj, ra = p.junction_capacity_j_k, p.case_capacity_j_k, p.junction_case_k_w, p.case_ambient_k_w
    a = np.array(
        [
            [-1 / (rj * cj), 1 / (rj * cj), power / cj, 0],
            [1 / (rj * cc), -(1 / rj + 1 / ra) / cc, ambient / (ra * cc), 0],
            [0, 0, 0, 0],
            [0, 1 / ra, -ambient / ra, 0],
        ]
    )
    result = expm(a * dt) @ np.r_[t, 1.0, 0.0]
    stored = float(np.array([cj, cc]) @ (result[:2] - t))
    return {
        "temperatures_c": result[:2],
        "input_j": power * dt,
        "ambient_j": float(result[3]),
        "stored_change_j": stored,
        "energy_residual_j": stored - power * dt + float(result[3]),
    }
