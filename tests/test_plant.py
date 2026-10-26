import numpy as np
import pytest

from hilconv import plant as m


def test_fault_splits_cycle_energy():
    from hilconv.faults import Fault, Schedule

    p = m.circuit.Circuit()
    x = np.array([4.0, 24.0])
    schedule = Schedule(p, [Fault(13e-6, "input_v", 36)])
    r = m.integrate_cycle(x, 0, 50e-6, 50e-6, "buck", 0.5, 0, schedule)
    assert 13e-6 in r["trace"][:, 0]
    assert m.circuit.energy(r["state"], p) - m.circuit.energy(x, p) == pytest.approx(
        r["input_j"] - r["output_j"] - r["loss_j"], abs=1e-12
    )


def test_short_interval_retains_carrier_phase():
    from hilconv.faults import Schedule

    p = m.circuit.Circuit()
    x = np.array([4.0, 24.0])
    schedule = Schedule(p)
    r = m.integrate_cycle(x, 0, 10e-6, 50e-6, "buck", 0.5, 0, schedule)
    expected = m.circuit.advance(x, 10e-6, 1, 1, p)
    assert np.allclose(r["state"], expected["state"])
    assert m.trace_output(r["trace"], schedule).shape == (2,)
