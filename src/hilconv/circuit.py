from dataclasses import dataclass

import numpy as np

from .core import finite, positive


@dataclass(frozen=True)
class Circuit:
    inductance_h: float = 220e-6
    capacitance_f: float = 470e-6
    inductor_ohm: float = 0.08
    capacitor_esr_ohm: float = 0.03
    input_v: float = 48
    load_ohm: float = 6
    load_source_v: float = 0

    def __post_init__(self):
        for key in ("inductance_h", "capacitance_f", "load_ohm"):
            positive(getattr(self, key))
        for key in ("inductor_ohm", "capacitor_esr_ohm"):
            positive(getattr(self, key), allow_zero=True)
        finite([self.input_v, self.load_source_v], real=True)


def state(value):
    x = finite(value, real=True)
    if x.shape != (2,):
        raise ValueError("inductor current and capacitor voltage required")
    return x


def ports(value, source_coefficient, output_coefficient, model=None):
    p = Circuit() if model is None else model
    x = state(value)
    sa, sb = finite([source_coefficient, output_coefficient], real=True)
    if not -1 <= sa <= 1 or not 0 <= sb <= 1:
        raise ValueError("bounded topology port coefficients required")
    current, capacitor = x
    denominator = 1 + p.capacitor_esr_ohm / p.load_ohm
    output = (
        capacitor + p.capacitor_esr_ohm * sb * current + p.capacitor_esr_ohm * p.load_source_v / p.load_ohm
    ) / denominator
    load = (output - p.load_source_v) / p.load_ohm
    cap = sb * current - load
    return {
        "output_v": float(output),
        "load_a": float(load),
        "capacitor_a": float(cap),
        "input_a": float(sa * current),
    }


def rhs(value, source_coefficient, output_coefficient, model=None):
    p = Circuit() if model is None else model
    x = state(value)
    r = ports(x, source_coefficient, output_coefficient, p)
    return np.array(
        [
            (source_coefficient * p.input_v - output_coefficient * r["output_v"] - p.inductor_ohm * x[0])
            / p.inductance_h,
            r["capacitor_a"] / p.capacitance_f,
        ]
    )


def energy(value, model=None):
    p = Circuit() if model is None else model
    x = state(value)
    return float(0.5 * p.inductance_h * x[0] ** 2 + 0.5 * p.capacitance_f * x[1] ** 2)


def power(value, sa, sb, model=None):
    p = Circuit() if model is None else model
    x = state(value)
    r = ports(x, sa, sb, p)
    return {
        "input_w": float(p.input_v * r["input_a"]),
        "output_w": float(r["output_v"] * r["load_a"]),
        "loss_w": float(p.inductor_ohm * x[0] ** 2 + p.capacitor_esr_ohm * r["capacitor_a"] ** 2),
    }


def generator(sa, sb, model=None):
    p = Circuit() if model is None else model
    zero = rhs([0, 0], sa, sb, p)
    matrix = np.column_stack([rhs([1, 0], sa, sb, p) - zero, rhs([0, 1], sa, sb, p) - zero, zero])
    return np.vstack([matrix, np.zeros(3)])
