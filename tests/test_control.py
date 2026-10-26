import numpy as np
import pytest

from hilconv import control as m


def test_pi_backcalculation():
    c = m.PI(0.1, 10, tracking_rate=2000)
    for _ in range(1000):
        c.update(100, 0.00005)
    assert abs(c.integral) < 10
    r = c.update(0, 0.00005)
    assert 0 <= r["command"] <= 1
    with pytest.raises(ValueError):
        c.update(1, 0)


def test_bumpless_pi():
    c = m.PI(0.2, 4)
    m.bumpless(c, 0.6, 2, 0.1)
    assert c.update(2, 0.001, 0.1)["command"] == pytest.approx(0.6)


def test_adc_rails_and_lsb():
    adc = m.ADC(8, 0, 50)
    r = adc.convert([-1, 0, 25, 50, 51])
    assert np.array_equal(r["code"], [0, 0, 128, 255, 255])
    assert np.array_equal(r["clipped"], [True, False, False, False, True])
    assert abs(r["value"][2] - 25) <= adc.lsb / 2
