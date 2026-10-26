from collections import deque
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


class Delay:
    def __init__(self, cycles=0, initial=0):
        integer(cycles, 0)
        self.queue = deque([initial] * cycles)

    def push(self, value):
        self.queue.append(value)
        return self.queue.popleft()


def timer_duty(duty, period_counts=1000):
    integer(period_counts, 2)
    value = finite(duty, real=True)
    if np.any((value < 0) | (value > 1)):
        raise ValueError("bounded PWM duty required")
    counts = np.rint(value * period_counts).astype(np.int64)
    return {"counts": counts, "duty": counts / period_counts}


def linearize(topology, duty_a, duty_b=0, model=None):
    from .pwm import averaged, equilibrium

    a, _ = averaged(topology, duty_a, duty_b, model)
    x = equilibrium(topology, duty_a, duty_b, model)
    delta = 1e-5
    if not delta < duty_a < 1 - delta:
        raise ValueError("interior duty operating point required")
    plus, _ = averaged(topology, duty_a + delta, duty_b, model)
    minus, _ = averaged(topology, duty_a - delta, duty_b, model)
    b = ((plus - minus) @ np.r_[x, 1.0])[:2] / (2 * delta)
    return {"state": x, "a": a[:2, :2], "b": b[:, None]}


def discrete_model(a, b, period_s):
    from scipy.linalg import expm

    a = finite(a, real=True, ndim=2)
    b = finite(b, real=True, ndim=2)
    n = len(a)
    if a.shape != (n, n) or b.shape[0] != n:
        raise ValueError("aligned continuous state matrices required")
    transition = expm(np.block([[a, b], [np.zeros((b.shape[1], n + b.shape[1]))]]) * positive(period_s))
    return transition[:n, :n], transition[:n, n:]


def lqr(a, b, state_cost, input_cost):
    from scipy.linalg import solve_discrete_are

    a, b, q, r = [finite(v, real=True, ndim=2) for v in (a, b, state_cost, input_cost)]
    if (
        not np.allclose(q, q.T)
        or not np.allclose(r, r.T)
        or min(np.linalg.eigvalsh(q)) < 0
        or min(np.linalg.eigvalsh(r)) <= 0
    ):
        raise ValueError("PSD state and positive definite input costs required")
    p = solve_discrete_are(a, b, q, r)
    gain = np.linalg.solve(r + b.T @ p @ b, b.T @ p @ a)
    return {"gain": gain, "riccati": p, "closed_loop_poles": np.linalg.eigvals(a - b @ gain)}


def deadtime_error(current_a, input_v, deadtime_s, switching_hz, bridge_legs=2):
    current = finite(current_a, real=True)
    vin = positive(input_v)
    dead = positive(deadtime_s, allow_zero=True)
    frequency = positive(switching_hz)
    integer(bridge_legs)
    if dead * frequency >= 0.5:
        raise ValueError("deadtime shorter than half a carrier period required")
    return -bridge_legs * vin * dead * frequency * np.sign(current)


def frequency_response(linearization, frequency_hz):
    f = finite(frequency_hz, real=True, ndim=1)
    if np.any(f <= 0):
        raise ValueError("positive response frequencies required")
    a = linearization["a"]
    b = linearization["b"]
    return np.array([(np.linalg.solve(2j * np.pi * v * np.eye(len(a)) - a, b))[1, 0] for v in f])


def observer_gain(a, measurement, poles):
    from scipy.signal import place_poles

    a = finite(a, real=True, ndim=2)
    h = finite(measurement, real=True, ndim=2)
    return place_poles(a.T, h.T, poles).gain_matrix.T


def observer_step(estimate, measurement, command, a, b, h, gain):
    x = finite(estimate, real=True, ndim=1)
    z = finite(measurement, real=True, ndim=1)
    u = np.atleast_1d(finite(command, real=True))
    return a @ x + b @ u + gain @ (z - h @ x)
