from itertools import pairwise

import numpy as np

from .core import finite, positive


def mode(topology, gate_a, gate_b=0):
    if gate_a not in (0, 1) or gate_b not in (0, 1):
        raise ValueError("binary gates required")
    if topology == "buck":
        return (gate_a, 1)
    if topology == "boost":
        return (1, 1 - gate_a)
    if topology == "bidirectional":
        return (gate_a, gate_b)
    if topology == "inverter":
        return (gate_a - gate_b, 1)
    raise ValueError("known converter topology required")


def segments(topology, duty_a, duty_b=0):
    da, db = finite([duty_a, duty_b], real=True)
    if not 0 <= da <= 1 or not 0 <= db <= 1:
        raise ValueError("duty fractions in [0,1] required")
    edges = sorted(set([0.0, float(da), float(db), 1.0]))
    return [
        (lo, hi, *mode(topology, int((lo + hi) / 2 < da), int((lo + hi) / 2 < db)))
        for lo, hi in pairwise(edges)
    ]


def fractions(topology, duty_a, duty_b=0):
    return [(hi - lo, sa, sb) for lo, hi, sa, sb in segments(topology, duty_a, duty_b)]


def shifted_segments(topology, duty_a, duty_b, phase_b):
    da, db, phase = finite([duty_a, duty_b, phase_b], real=True)
    if not 0 <= da <= 1 or not 0 <= db <= 1 or not 0 <= phase < 1:
        raise ValueError("bounded duties and wrapped phase required")
    edges = sorted(set([0.0, 1.0, float(da), float(phase), float((phase + db) % 1)]))
    result = []
    for lo, hi in pairwise(edges):
        mid = (lo + hi) / 2
        sa, sb = mode(topology, int(mid < da), int(((mid - phase) % 1) < db))
        result.append((lo, hi, sa, sb))
    return result


def cycle(value, period_s, topology, duty_a, duty_b=0, model=None):
    from .circuit import advance, state

    period = positive(period_s)
    x = state(value)
    totals = {"input_j": 0.0, "output_j": 0.0, "loss_j": 0.0}
    for lo, hi, sa, sb in segments(topology, duty_a, duty_b):
        r = advance(x, (hi - lo) * period, sa, sb, model)
        x = r["state"]
        for key in totals:
            totals[key] += r[key]
    return {"state": x, **totals}


def averaged(topology, duty_a, duty_b=0, model=None):
    from .circuit import generator, power_forms

    parts = fractions(topology, duty_a, duty_b)
    a = sum(f * generator(sa, sb, model) for f, sa, sb in parts)
    q = sum(f * power_forms(sa, sb, model) for f, sa, sb in parts)
    return a, q


def average_step(value, dt_s, topology, duty_a, duty_b=0, model=None):
    from .circuit import operators, state

    a, q = averaged(topology, duty_a, duty_b, model)
    z = np.r_[state(value), 1.0]
    transition, integrals = operators(tuple(a.ravel()), tuple(q.ravel()), dt_s)
    energy = integrals @ np.outer(z, z).ravel()
    return {
        "state": (transition @ z)[:2],
        "input_j": float(energy[0]),
        "output_j": float(energy[1]),
        "loss_j": float(energy[2]),
    }


def equilibrium(topology, duty_a, duty_b=0, model=None):
    a, _ = averaged(topology, duty_a, duty_b, model)
    if np.linalg.cond(a[:2, :2]) > 1e12:
        raise ValueError("finite nonsingular operating point required")
    return np.linalg.solve(a[:2, :2], -a[:2, 2])
