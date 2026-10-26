import pytest

from hilconv import faults as m


def test_fault_validation():
    assert m.Fault(0.01, "load_ohm", 0.1).parameter == "load_ohm"
    with pytest.raises(ValueError):
        m.Fault(0.01, "capacitance_f", 0)
    with pytest.raises(ValueError):
        m.Fault(0.01, "load_ohm", 0)
