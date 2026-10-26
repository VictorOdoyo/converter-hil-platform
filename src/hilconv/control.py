from dataclasses import dataclass

import numpy as np

from .core import finite, integer, positive


class PI:
    def __init__(self, kp, ki, lower=0, upper=1, tracking_rate=1000):
        self.kp = positive(kp, allow_zero=True)
        self.ki = positive(ki, allow_zero=True)
        self.tracking = positive(tracking_rate, allow_zero=True)
        self.lower, self.upper = finite([lower, upper], real=True)
        if self.lower >= self.upper:
            raise ValueError("ordered actuator limits required")
        self.integral = 0.0

    def update(self, error, dt_s, feedforward=0):
        error = float(finite(error, real=True))
        dt = positive(dt_s)
        ff = float(finite(feedforward, real=True))
        raw = ff + self.kp * error + self.integral
        command = float(np.clip(raw, self.lower, self.upper))
        correction = self.tracking * (command - raw)
        self.integral += dt * (self.ki * error + correction)
        return {
            "command": command,
            "unsaturated": raw,
            "integral": self.integral,
            "saturated": bool(command != raw),
        }


def bumpless(controller, command, error=0, feedforward=0):
    command = float(finite(command, real=True))
    if not controller.lower <= command <= controller.upper:
        raise ValueError("initial command inside actuator limits required")
    controller.integral = (
        command - controller.kp * float(finite(error, real=True)) - float(finite(feedforward, real=True))
    )
    return controller.integral


@dataclass(frozen=True)
class ADC:
    bits: int = 12
    lower: float = -20
    upper: float = 20
    offset: float = 0
    gain: float = 1

    def __post_init__(self):
        integer(self.bits)
        finite([self.lower, self.upper, self.offset, self.gain], real=True)
        if self.bits > 24 or self.lower >= self.upper or self.gain <= 0:
            raise ValueError("bounded ADC range and positive sensor gain required")

    @property
    def lsb(self):
        return (self.upper - self.lower) / (2**self.bits - 1)

    def convert(self, value):
        raw = finite(value, real=True) * self.gain + self.offset
        clipped = np.clip(raw, self.lower, self.upper)
        code = np.rint((clipped - self.lower) / self.lsb).astype(np.int64)
        return {
            "code": code,
            "value": self.lower + code * self.lsb,
            "clipped": (raw < self.lower) | (raw > self.upper),
        }
