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
