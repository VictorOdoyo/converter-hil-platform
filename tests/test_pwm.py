import numpy as np
import pytest

from hilconv import pwm as m


def test_topology_modes():
    assert m.mode("buck", 0) == (0, 1)
    assert m.mode("boost", 1) == (1, 0)
    assert m.mode("inverter", 0, 1) == (-1, 1)
    with pytest.raises(ValueError):
        m.mode("unknown", 1)


def test_common_carrier_overlap():
    parts = m.fractions("bidirectional", 0.7, 0.4)
    assert np.allclose(parts, [(0.4, 1, 1), (0.3, 1, 0), (0.3, 0, 0)])
    assert m.fractions("buck", 0) == [(1.0, 0, 1)]
    assert m.fractions("buck", 1) == [(1.0, 1, 1)]


def test_shifted_intervals():
    parts = m.shifted_segments("inverter", 0.5, 0.5, 0.5)
    assert parts == [(0.0, 0.5, 1, 1), (0.5, 1.0, -1, 1)]
    assert sum(hi - lo for lo, hi, _, _ in parts) == 1


def test_cycle_energy():
    from hilconv.circuit import energy

    x = np.array([3.0, 24.0])
    r = m.cycle(x, 50e-6, "buck", 0.5)
    assert energy(r["state"]) - energy(x) == pytest.approx(
        r["input_j"] - r["output_j"] - r["loss_j"], abs=1e-12
    )


def test_averaged_esr_not_duty_substitution():
    from hilconv.circuit import Circuit, energy, generator

    p = Circuit(capacitor_esr_ohm=0.5)
    a, _q = m.averaged("boost", 0.5, model=p)
    assert not np.allclose(a, generator(1, 0.5, p))
    x = np.array([5.0, 70.0])
    r = m.average_step(x, 0.0001, "boost", 0.5, model=p)
    assert energy(r["state"], p) - energy(x, p) == pytest.approx(
        r["input_j"] - r["output_j"] - r["loss_j"], abs=2e-12
    )


def test_ideal_steady_conversion_ratios():
    from hilconv.circuit import Circuit

    p = Circuit(inductor_ohm=0, capacitor_esr_ohm=0)
    assert m.equilibrium("buck", 0.4, model=p)[1] == pytest.approx(0.4 * 48)
    assert m.equilibrium("boost", 0.4, model=p)[1] == pytest.approx(48 / 0.6)
    assert m.equilibrium("inverter", 0.75, 0.25, p)[1] == pytest.approx(24)


def test_switched_periodic_fixed_point():
    x = m.periodic_state(50e-6, "boost", 0.4)
    y = m.cycle(x, 50e-6, "boost", 0.4)["state"]
    assert np.allclose(x, y, atol=1e-10)


def test_waveform_end_matches_cycle():
    x = m.periodic_state(50e-6, "buck", 0.5)
    r = m.waveform(x, 50e-6, "buck", 0.5)
    assert np.allclose(r["state"][-1], m.cycle(x, 50e-6, "buck", 0.5)["state"])
    assert np.ptp(r["state"][:, 0]) > 2
