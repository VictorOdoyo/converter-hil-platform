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
