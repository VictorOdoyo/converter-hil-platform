from itertools import pairwise

import numpy as np

from .core import finite, integer, positive


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


def periodic_state(period_s, topology, duty_a, duty_b=0, model=None):
    from scipy.linalg import expm

    from .circuit import generator

    period = positive(period_s)
    mapping = np.eye(3)
    for lo, hi, sa, sb in segments(topology, duty_a, duty_b):
        mapping = expm(generator(sa, sb, model) * (hi - lo) * period) @ mapping
    matrix = np.eye(2) - mapping[:2, :2]
    if np.linalg.cond(matrix) > 1e12:
        raise ValueError("nonsingular periodic operating point required")
    return np.linalg.solve(matrix, mapping[:2, 2])


def waveform(value, period_s, topology, duty_a, duty_b=0, model=None, samples=101):
    from scipy.linalg import expm

    from .circuit import generator, ports, state

    integer(samples, 2)
    period = positive(period_s)
    t = np.linspace(0, period, samples)
    initial = state(value)
    parts = segments(topology, duty_a, duty_b)
    states = []
    output = []
    mode_a = []
    mode_b = []
    for time in t:
        x = initial.copy()
        phase = min(time / period, np.nextafter(1.0, 0.0))
        active = parts[-1][2:]
        for lo, hi, sa, sb in parts:
            if time >= lo * period:
                duration = min(time, hi * period) - lo * period
                if duration > 0:
                    x = (expm(generator(sa, sb, model) * duration) @ np.r_[x, 1.0])[:2]
            if lo <= phase < hi:
                active = (sa, sb)
        states.append(x)
        output.append(ports(x, *active, model)["output_v"])
        mode_a.append(active[0])
        mode_b.append(active[1])
    return {
        "time_s": t,
        "state": np.array(states),
        "output_v": np.array(output),
        "source_coefficient": np.array(mode_a),
        "output_coefficient": np.array(mode_b),
    }


def buck_ripple(input_v, output_v, duty, switching_hz, inductance_h, capacitance_f, esr_ohm=0):
    vin = positive(input_v)
    vout = positive(output_v)
    d = positive(duty)
    f = positive(switching_hz)
    ind = positive(inductance_h)
    cap = positive(capacitance_f)
    esr = positive(esr_ohm, allow_zero=True)
    if d >= 1 or vout >= vin or not np.isclose(vout, vin * d):
        raise ValueError("ideal steady CCM buck conversion ratio required")
    ripple = (vin - vout) * d / (ind * f)
    return {
        "inductor_peak_to_peak_a": ripple,
        "capacitive_peak_to_peak_v": ripple / (8 * f * cap),
        "esr_peak_to_peak_v": esr * ripple,
    }
