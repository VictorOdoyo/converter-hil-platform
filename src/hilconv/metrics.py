import numpy as np

from .core import finite, integer, positive, timebase


def statistics(time_s, values):
    t = timebase(time_s)
    y = finite(values, real=True, ndim=1)
    if y.shape != t.shape:
        raise ValueError("aligned physical time samples required")
    span = t[-1] - t[0]
    mean = np.trapezoid(y, t) / span
    rms = np.sqrt(np.trapezoid(y * y, t) / span)
    return {"mean": float(mean), "rms": float(rms), "peak_to_peak": float(np.ptp(y))}


def harmonics(time_s, values, fundamental_hz=50, maximum_harmonic=20):
    t = timebase(time_s)
    y = finite(values, real=True, ndim=1)
    f = positive(fundamental_hz)
    integer(maximum_harmonic)
    if y.shape != t.shape or len(t) < 2 * maximum_harmonic + 2:
        raise ValueError("sufficient aligned harmonic samples required")
    phase = 2 * np.pi * f * t
    columns = [np.ones(len(t))]
    for k in range(1, maximum_harmonic + 1):
        columns.extend([np.cos(k * phase), np.sin(k * phase)])
    matrix = np.column_stack(columns)
    coeff = np.linalg.lstsq(matrix, y, rcond=None)[0]
    if np.linalg.matrix_rank(matrix) < matrix.shape[1]:
        raise ValueError("unobservable harmonic design matrix")
    amplitude = np.hypot(coeff[1::2], coeff[2::2])
    fundamental = amplitude[0]
    return {
        "dc": float(coeff[0]),
        "amplitude": amplitude,
        "thd": float(np.linalg.norm(amplitude[1:]) / fundamental) if fundamental > 1e-12 else 0.0,
        "fundamental_observable": bool(fundamental > 1e-12),
        "residual_rms": float(np.sqrt(np.mean((y - matrix @ coeff) ** 2))),
    }


def settling(time_s, values, target, tolerance):
    t = timebase(time_s)
    y = finite(values, real=True, ndim=1)
    tol = positive(tolerance)
    if y.shape != t.shape:
        raise ValueError("aligned settling samples required")
    outside = abs(y - float(finite(target, real=True))) > tol
    if outside[-1]:
        return {"settled": False, "time_s": None}
    bad = np.flatnonzero(outside)
    index = 0 if not len(bad) else int(bad[-1] + 1)
    return {"settled": True, "time_s": float(t[index])}
