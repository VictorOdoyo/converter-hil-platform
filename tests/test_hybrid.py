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
