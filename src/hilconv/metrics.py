import numpy as np

from .core import finite, timebase


def statistics(time_s, values):
    t = timebase(time_s)
    y = finite(values, real=True, ndim=1)
    if y.shape != t.shape:
        raise ValueError("aligned physical time samples required")
    span = t[-1] - t[0]
    mean = np.trapezoid(y, t) / span
    rms = np.sqrt(np.trapezoid(y * y, t) / span)
    return {"mean": float(mean), "rms": float(rms), "peak_to_peak": float(np.ptp(y))}
