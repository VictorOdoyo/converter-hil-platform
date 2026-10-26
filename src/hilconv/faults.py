from dataclasses import dataclass, replace

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


class Schedule:
    def __init__(self, model, events=()):
        self.initial = model
        self.events = tuple(sorted(events, key=lambda f: f.time_s))
        if any(not isinstance(f, Fault) for f in self.events):
            raise ValueError("validated physical faults required")

    def at(self, time_s):
        time = float(finite(time_s, real=True))
        p = self.initial
        for event in self.events:
            if event.time_s > time + 1e-14:
                break
            p = replace(p, **{event.parameter: event.value})
        return p

    def inside(self, start_s, stop_s):
        if stop_s <= start_s:
            raise ValueError("positive integration interval required")
        return sorted({f.time_s for f in self.events if start_s < f.time_s < stop_s})
