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


def diode_advance(value, duration_s, sa, sb, model=None):
    p = Circuit() if model is None else model
    x = state(value)
    remaining = positive(duration_s)
    totals = {"input_j": 0.0, "output_j": 0.0, "loss_j": 0.0}
    blocked = 0.0
    events = 0
    if x[0] < -1e-10 or sa not in (0, 1) or sb not in (0, 1) or min(p.input_v, p.load_source_v) < 0:
        raise ValueError("nonnegative diode-port operating state required")
    for _ in range(64):
        if remaining < 1e-15:
            break
        blocking = (
            sb == 1 and x[0] <= 1e-10 and ports([0, x[1]], sa, sb, p)["output_v"] > sa * p.input_v + 1e-10
        )
        if blocking:
            x[0] = 0.0
            dt = idle_until_restart(x, remaining, sa, p)
            r = advance(x, dt, 0, 0, p)
            blocked += dt
        else:
            crossing = first_current_crossing(x, remaining, sa, sb, model=p) if sb else None
            dt = remaining if crossing is None else crossing
            r = advance(x, dt, sa, sb, p)
            if crossing is not None:
                r["state"][0] = 0.0
                events += 1
        x = r["state"]
        remaining -= dt
        for key in totals:
            totals[key] += r[key]
    else:
        raise RuntimeError("diode event limit exceeded")
    return {"state": x, **totals, "blocked_s": blocked, "zero_crossings": events}


def diode_cycle(value, period_s, topology, duty, model=None):
    from .pwm import segments

    if topology not in ("buck", "boost"):
        raise ValueError("unidirectional buck or boost required")
    x = state(value)
    totals = {"input_j": 0.0, "output_j": 0.0, "loss_j": 0.0, "blocked_s": 0.0, "zero_crossings": 0}
    for lo, hi, sa, sb in segments(topology, duty):
        r = diode_advance(x, (hi - lo) * positive(period_s), sa, sb, model)
        x = r["state"]
        for key in totals:
            totals[key] += r[key]
    return {"state": x, **totals}
