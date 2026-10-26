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
    trace = [(0.0, float(x[0]), float(x[1]))]
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
        trace.append((duration_s - remaining, float(x[0]), float(x[1])))
        for key in totals:
            totals[key] += r[key]
    else:
        raise RuntimeError("diode event limit exceeded")
    return {"state": x, **totals, "blocked_s": blocked, "zero_crossings": events, "trace": np.array(trace)}


def diode_cycle(value, period_s, topology, duty, model=None):
    from .pwm import segments

    if topology not in ("buck", "boost"):
        raise ValueError("unidirectional buck or boost required")
    x = state(value)
    totals = {"input_j": 0.0, "output_j": 0.0, "loss_j": 0.0, "blocked_s": 0.0, "zero_crossings": 0}
    trace = [(0.0, float(x[0]), float(x[1]))]
    for lo, hi, sa, sb in segments(topology, duty):
        r = diode_advance(x, (hi - lo) * positive(period_s), sa, sb, model)
        trace.extend((lo * period_s + row[0], row[1], row[2]) for row in r["trace"][1:])
        x = r["state"]
        for key in totals:
            totals[key] += r[key]
    return {"state": x, **totals, "trace": np.array(trace)}


def limited_buck_cycle(value, period_s, duty, current_limit_a, model=None):
    from .core import positive

    p = Circuit() if model is None else model
    period = positive(period_s)
    limit = positive(current_limit_a)
    if not 0 <= duty <= 1:
        raise ValueError("bounded duty required")
    x = state(value)
    on = period * duty
    if x[0] >= limit:
        on = 0.0
    elif on > 0:
        crossing = first_current_crossing(x, on, 1, 1, limit, 1, p)
        if crossing is not None:
            on = crossing
    first = advance(x, on, 1, 1, p)
    second = advance(first["state"], period - on, 0, 1, p)
    return {
        "state": second["state"],
        **{key: first[key] + second[key] for key in ("input_j", "output_j", "loss_j")},
        "effective_duty": on / period,
        "limited": bool(on < period * duty - 1e-15),
        "turnoff_a": float(first["state"][0]),
    }
