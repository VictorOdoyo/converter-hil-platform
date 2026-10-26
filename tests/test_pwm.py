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
