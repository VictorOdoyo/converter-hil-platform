from dataclasses import dataclass

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
