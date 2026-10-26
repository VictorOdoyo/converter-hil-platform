from itertools import pairwise

import numpy as np

from . import circuit, pwm
from .core import finite, positive


def integrate_cycle(
    value,
    start_s,
    duration_s,
    period_s,
    topology,
    duty_a,
    duty_b,
    schedule,
    averaged=False,
    diode=False,
    current_limit_a=None,
):
    from .hybrid import diode_advance, first_current_crossing

    start = float(finite(start_s, real=True))
    duration = positive(duration_s)
    period = positive(period_s)
    if duration > period * (1 + 1e-10):
        raise ValueError("cycle slice cannot exceed carrier period")
    if diode and (averaged or topology not in ("buck", "boost")):
        raise ValueError("diode events require switched buck or boost")
    x = circuit.state(value)
    stop = start + duration
    parts = pwm.segments(topology, duty_a, duty_b)
    edges = {start, stop, *schedule.inside(start, stop)}
    if not averaged:
        edges.update(start + hi * period for _, hi, _, _ in parts if start < start + hi * period < stop)
    totals = {"input_j": 0.0, "output_j": 0.0, "loss_j": 0.0, "blocked_s": 0.0}
    limited = False
    turned_off = False
    trace = [(start, float(x[0]), float(x[1]))]
    for lo, hi in pairwise(sorted(edges)):
        p = schedule.at(lo)
        dt = hi - lo
        if averaged:
            r = pwm.average_step(x, dt, topology, duty_a, duty_b, p)
        else:
            phase = ((lo + hi) / 2 - start) / period
            sa, sb = next((a, b) for left, right, a, b in parts if left <= phase < right + 1e-14)
            if topology == "buck" and current_limit_a is not None:
                limit = positive(current_limit_a)
                if x[0] >= limit:
                    turned_off = True
                if turned_off:
                    sa = 0
                if sa == 1:
                    crossing = first_current_crossing(x, dt, sa, sb, limit, 1, p)
                    if crossing is not None:
                        first = circuit.advance(x, crossing, sa, sb, p)
                        tail = circuit.advance(first["state"], dt - crossing, 0, 1, p)
                        r = {
                            "state": tail["state"],
                            **{key: first[key] + tail[key] for key in ("input_j", "output_j", "loss_j")},
                        }
                        trace.append((lo + crossing, float(first["state"][0]), float(first["state"][1])))
                        turned_off = True
                        limited = True
                    else:
                        r = circuit.advance(x, dt, sa, sb, p)
                else:
                    r = circuit.advance(x, dt, sa, sb, p)
            elif diode:
                r = diode_advance(x, dt, sa, sb, p)
            else:
                r = circuit.advance(x, dt, sa, sb, p)
        x = r["state"]
        for key in totals:
            totals[key] += r.get(key, 0.0)
        trace.append((hi, float(x[0]), float(x[1])))
    return {"state": x, **totals, "limited": limited, "trace": np.array(trace)}
