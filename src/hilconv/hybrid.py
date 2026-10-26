from itertools import pairwise

import numpy as np

from .circuit import Circuit, advance, generator, ports, state
from .core import positive


def first_current_crossing(value, duration_s, sa, sb, level_a=0, direction=-1, model=None):
    from scipy.linalg import expm
    from scipy.optimize import brentq

    p = Circuit() if model is None else model
    x = state(value)
    duration = positive(duration_s)
    if direction not in (-1, 1):
        raise ValueError("crossing direction required")
    a = generator(sa, sb, p)
    z = np.r_[x, 1.0]

    def residual(time):
        return float((expm(a * time) @ z)[0] - level_a)

    count = max(8, int(np.ceil(duration / (np.sqrt(p.inductance_h * p.capacitance_f) / 16))))
    grid = np.linspace(0, duration, count + 1)
    previous = residual(0)
    for left, right in pairwise(grid):
        value = residual(right)
        if direction * previous < 0 and direction * value >= 0:
            return float(brentq(residual, left, right, xtol=1e-15, rtol=1e-13))
        previous = value
    return None


def idle_until_restart(value, duration_s, sa, model=None):
    from scipy.optimize import brentq

    p = Circuit() if model is None else model
    x = state(value)
    duration = positive(duration_s)
    if abs(x[0]) > 1e-8:
        raise ValueError("zero inductor current required for diode blocking")

    def margin(time):
        y = advance(x, time, 0, 0, p)["state"]
        return ports(y, 0, 0, p)["output_v"] - sa * p.input_v

    if margin(0) <= 0:
        return 0.0
    if margin(duration) >= 0:
        return duration
    return float(brentq(margin, 0, duration, xtol=1e-15))
