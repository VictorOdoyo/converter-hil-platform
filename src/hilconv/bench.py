from dataclasses import asdict, dataclass

from .core import finite, integer, positive


@dataclass(frozen=True)
class Config:
    duration_s: float = 0.04
    switching_hz: float = 20000
    reference_v: float = 24
    reference_step_v: float = 24
    reference_step_s: float = 0.025
    softstart_s: float = 0.003
    adc_bits: int = 12
    voltage_bias_v: float = 0
    measurement_delay: int = 0
    latency_ns: int = 50000
    jitter_ns: int = 0
    drop_every: int = 0
    corrupt_every: int = 0
    blackout_s: tuple = ()
    pwm_counts: int = 1000
    antiwindup: bool = True
    averaged: bool = False
    current_limit_a: float = 15
    seed: int = 42

    def __post_init__(self):
        for key in (
            "duration_s",
            "switching_hz",
            "reference_v",
            "reference_step_v",
            "softstart_s",
            "current_limit_a",
        ):
            positive(getattr(self, key))
        positive(self.reference_step_s, allow_zero=True)
        for key in ("measurement_delay", "latency_ns", "jitter_ns", "drop_every", "corrupt_every", "seed"):
            integer(getattr(self, key), 0)
        integer(self.adc_bits)
        integer(self.pwm_counts, 2)
        finite(self.voltage_bias_v, real=True)
        if (
            self.adc_bits > 24
            or self.seed >= 2**32
            or type(self.averaged) is not bool
            or type(self.antiwindup) is not bool
        ):
            raise ValueError("bounded resolution seed and boolean model switches required")
        if self.blackout_s:
            finite(self.blackout_s, real=True)
            if len(self.blackout_s) != 2 or not 0 <= self.blackout_s[0] < self.blackout_s[1]:
                raise ValueError("ordered blackout interval required")

    def to_dict(self):
        return asdict(self)
