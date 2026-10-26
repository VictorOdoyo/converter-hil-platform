import numpy as np
import pytest

from hilconv import hybrid as m


def test_current_zero_event():
    from hilconv.circuit import Circuit, advance

    p = Circuit(inductor_ohm=0, capacitor_esr_ohm=0)
    t = m.first_current_crossing([1, 24], 50e-6, 0, 1, model=p)
    assert 8e-6 < t < 10e-6
    assert abs(advance([1, 24], t, 0, 1, p)["state"][0]) < 1e-9


def test_boost_diode_restart():
    p = m.Circuit(input_v=48, load_ohm=6)
    t = m.idle_until_restart([0, 55], 0.01, 1, p)
    assert 0 < t < 0.01
    x = m.advance([0, 55], t, 0, 0, p)["state"]
    assert m.ports(x, 0, 0, p)["output_v"] == pytest.approx(48, abs=1e-8)


def test_dcm_balance_no_negative_current():
    from hilconv.circuit import energy

    p = m.Circuit()
    x = np.array([1.0, 24.0])
    r = m.diode_advance(x, 50e-6, 0, 1, p)
    assert r["state"][0] == 0
    assert r["blocked_s"] > 30e-6
    assert r["zero_crossings"] == 1
    assert energy(r["state"], p) - energy(x, p) == pytest.approx(
        r["input_j"] - r["output_j"] - r["loss_j"], abs=1e-12
    )


def test_light_load_dcm_cycle():
    p = m.Circuit(load_ohm=100)
    r = m.diode_cycle([0, 24], 50e-6, "buck", 0.1, p)
    assert r["state"][0] >= 0
    assert r["blocked_s"] > 0


def test_comparator_limits_on_pulse():
    r = m.limited_buck_cycle([2, 24], 50e-6, 0.9, 4)
    assert r["limited"]
    assert r["turnoff_a"] == pytest.approx(4, abs=1e-8)
    assert 0 < r["effective_duty"] < 0.9


def test_boost_idle_restarts_with_energy_continuity():
    from hilconv.circuit import energy

    p = m.Circuit(input_v=48, load_ohm=6)
    x = np.array([0.0, 55.0])
    r = m.diode_advance(x, 0.001, 1, 1, p)
    assert r["blocked_s"] > 0 and r["blocked_s"] < 0.001
    assert r["state"][0] > 0
    assert energy(r["state"], p) - energy(x, p) == pytest.approx(
        r["input_j"] - r["output_j"] - r["loss_j"], abs=1e-9
    )


def test_zero_event_trace_includes_blocked_tail():
    r = m.diode_advance([1, 24], 50e-6, 0, 1)
    assert r["trace"].shape == (3, 3)
    assert 0 < r["trace"][1, 0] < 50e-6
    assert r["trace"][1, 1] == 0 and r["trace"][-1, 1] == 0
    assert r["trace"][-1, 0] == pytest.approx(50e-6)
