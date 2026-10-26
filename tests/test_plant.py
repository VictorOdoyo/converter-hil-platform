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


def test_roundoff_at_final_carrier_edge():
    from hilconv.faults import Schedule

    p = m.circuit.Circuit()
    r = m.integrate_cycle([3, 24], 0.00415, 5.0000000000000565e-5, 5e-5, "buck", 0.556, 0, Schedule(p))
    assert len(r["trace"]) == 3
    assert np.all(np.diff(r["trace"][:, 0]) > 0)


def test_already_over_limit_reports_inhibition():
    from hilconv.faults import Schedule

    p = m.circuit.Circuit()
    r = m.integrate_cycle([16, 24], 0, 50e-6, 50e-6, "buck", 0.5, 0, Schedule(p), current_limit_a=15)
    assert r["limited"]
    assert r["input_j"] == 0


def test_plant_diode_trace_retains_zero_crossing():
    from hilconv.faults import Schedule

    p = m.circuit.Circuit()
    r = m.integrate_cycle([0, 24], 0, 50e-6, 50e-6, "buck", 0.15, 0, Schedule(p), diode=True)
    assert len(r["trace"]) == 4
    assert max(r["trace"][:, 1]) > 0
    assert r["trace"][-2, 1] == 0 and r["trace"][-1, 1] == 0
