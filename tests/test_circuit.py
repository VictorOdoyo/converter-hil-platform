import numpy as np
import pytest

from hilconv import circuit as m


def test_circuit_passivity():
    assert m.Circuit().input_v == 48
    with pytest.raises(ValueError):
        m.Circuit(capacitor_esr_ohm=-1)
    with pytest.raises(ValueError):
        m.state([1, 2, 3])


def test_kcl_esr():
    p = m.Circuit(capacitor_esr_ohm=0.2, load_source_v=10)
    r = m.ports([4, 20], 1, 1, p)
    assert 4 == pytest.approx(r["load_a"] + r["capacitor_a"])
    assert r["output_v"] - 20 == pytest.approx(0.2 * r["capacitor_a"])


def test_ideal_buck_slopes():
    p = m.Circuit(inductor_ohm=0, capacitor_esr_ohm=0)
    assert m.rhs([0, 24], 1, 1, p)[0] == pytest.approx(24 / p.inductance_h)
    assert m.rhs([0, 24], 0, 1, p)[0] == pytest.approx(-24 / p.inductance_h)


def test_instantaneous_energy_balance():
    p = m.Circuit(load_source_v=24)
    for sa, sb in [(1, 1), (1, 0), (0, 1), (-1, 1)]:
        x = np.array([-2.0, 18.0])
        derivative = m.rhs(x, sa, sb, p)
        w = m.power(x, sa, sb, p)
        de = p.inductance_h * x[0] * derivative[0] + p.capacitance_f * x[1] * derivative[1]
        assert de == pytest.approx(w["input_w"] - w["output_w"] - w["loss_w"], abs=1e-10)
