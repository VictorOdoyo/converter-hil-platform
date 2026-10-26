import numpy as np

from .core import finite, positive


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
