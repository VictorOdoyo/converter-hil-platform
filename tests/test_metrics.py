import numpy as np
import pytest

from hilconv import metrics as m


def test_time_weighting():
    r = m.statistics([0, 1, 3], [0, 2, 2])
    assert r["mean"] == pytest.approx(5 / 3)
    assert r["peak_to_peak"] == 2


def test_harmonic_fit():
    t = np.linspace(0.013, 0.093, 1000)
    y = 2 + 10 * np.sin(2 * np.pi * 50 * t) + np.sin(2 * np.pi * 150 * t)
    r = m.harmonics(t, y, 50, 5)
    assert r["dc"] == pytest.approx(2)
    assert r["thd"] == pytest.approx(0.1)
    assert r["residual_rms"] < 1e-10


def test_settling_retains_failure():
    assert not m.settling([0, 1, 2], [0, 1, 0.5], 1, 0.1)["settled"]
    assert m.settling([0, 1, 2], [0, 1, 1], 1, 0.1)["time_s"] == 1
