from itertools import pairwise

from .core import finite


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
