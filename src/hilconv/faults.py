from dataclasses import dataclass

from .core import finite, positive


@dataclass(frozen=True)
class Fault:
    time_s: float
    parameter: str
    value: float

    def __post_init__(self):
        positive(self.time_s, allow_zero=True)
        finite(self.value, real=True)
        if self.parameter not in (
            "input_v",
            "load_ohm",
            "load_source_v",
            "inductor_ohm",
            "capacitor_esr_ohm",
        ):
            raise ValueError("fault must change a physical non-storage parameter")
        if self.parameter == "load_ohm":
            positive(self.value)
        if self.parameter in ("inductor_ohm", "capacitor_esr_ohm"):
            positive(self.value, allow_zero=True)
