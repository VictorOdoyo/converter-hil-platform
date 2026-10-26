import numpy as np


def finite(value, *, real=False, ndim=None):
    array = np.asarray(value)
    if array.dtype.kind not in "biufc" or not np.all(np.isfinite(array)):
        raise ValueError("finite numeric values required")
    if real and np.iscomplexobj(array):
        raise ValueError("real values required")
    if ndim is not None and array.ndim != ndim:
        raise ValueError("incorrect array rank")
    return array.astype(float if real else complex)


def positive(value, name="parameter", allow_zero=False):
    if not np.isscalar(value) or np.iscomplexobj(value) or isinstance(value, (bool, np.bool_)):
        raise ValueError(name + " must be a finite real scalar")
    value = float(value)
    if not np.isfinite(value) or (value < 0 if allow_zero else value <= 0):
        raise ValueError(name + " must be positive" if not allow_zero else name + " must be nonnegative")
    return value


def integer(value, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError("integer count required")
    return value


def stream(seed, name):
    import hashlib

    integer(seed, 0)
    if seed >= 2**32 or not isinstance(name, str):
        raise ValueError("32-bit seed and named stream required")
    key = int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], "little")
    return np.random.default_rng(np.random.SeedSequence([seed, key]))


def timebase(time):
    time = finite(time, real=True, ndim=1)
    if len(time) < 2 or np.any(np.diff(time) <= 0):
        raise ValueError("at least two strictly increasing timestamps required")
    return time


def integrate(time, values):
    time = timebase(time)
    values = finite(values)
    if values.shape[0] != len(time):
        raise ValueError("aligned timestamps required")
    return np.trapezoid(values, time, axis=0)


def timeline(duration_s, maximum_step_s):
    duration = positive(duration_s)
    step = positive(maximum_step_s)
    count = int(np.floor(duration / step))
    values = np.arange(count + 1) * step
    if duration - values[-1] > 1e-12 * max(1.0, duration):
        values = np.r_[values, duration]
    else:
        values[-1] = duration
    if len(values) < 2:
        values = np.array([0.0, duration])
    return values
