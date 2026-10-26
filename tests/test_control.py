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


def test_delay_is_causal():
    delay = m.Delay(2, -1)
    assert [delay.push(v) for v in [10, 20, 30, 40]] == [-1, -1, 10, 20]
    assert m.Delay(0).push(9) == 9


def test_timer_endpoints():
    r = m.timer_duty([0, 0.1234, 1], 1000)
    assert np.array_equal(r["counts"], [0, 123, 1000])
    with pytest.raises(ValueError):
        m.timer_duty(1.1)


def test_discrete_linearization():
    r = m.linearize("buck", 0.5)
    a, b = m.discrete_model(r["a"], r["b"], 50e-6)
    assert a.shape == (2, 2) and b.shape == (2, 1)
    assert max(abs(np.linalg.eigvals(a))) < 1


def test_lqr_stability():
    lin = m.linearize("buck", 0.5)
    a, b = m.discrete_model(lin["a"], lin["b"], 50e-6)
    result = m.lqr(a, b, np.diag([0.01, 1]), np.array([[20.0]]))
    assert max(abs(result["closed_loop_poles"])) < 1


def test_deadtime_polarity():
    assert np.allclose(m.deadtime_error([-2, 0, 2], 48, 1e-6, 20000), [1.92, 0, -1.92])


def test_boost_rhp_zero_response():
    from scipy.signal import ss2tf

    from hilconv.circuit import Circuit

    p = Circuit(inductor_ohm=0, capacitor_esr_ohm=0)
    lin = m.linearize("boost", 0.4, model=p)
    numerator, _ = ss2tf(lin["a"], lin["b"], np.array([[0.0, 1.0]]), np.zeros((1, 1)))
    roots = np.roots(numerator[0])
    assert np.any(np.isclose(roots, p.load_ohm * 0.6**2 / p.inductance_h, rtol=1e-6))
    assert abs(m.frequency_response(lin, [1])[0]) > 100


def test_voltage_only_observer_converges():
    lin = m.linearize("buck", 0.5)
    a, b = m.discrete_model(lin["a"], lin["b"], 50e-6)
    h = np.array([[0.0, 1.0]])
    gain = m.observer_gain(a, h, [0.7, 0.8])
    truth = np.array([1.0, 2.0])
    estimate = np.zeros(2)
    for _ in range(100):
        estimate = m.observer_step(estimate, h @ truth, [0], a, b, h, gain)
        truth = a @ truth
    assert np.linalg.norm(estimate - truth) < 1e-7
