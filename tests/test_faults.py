import pytest

from hilconv import faults as m


def test_fault_validation():
    assert m.Fault(0.01, "load_ohm", 0.1).parameter == "load_ohm"
    with pytest.raises(ValueError):
        m.Fault(0.01, "capacitance_f", 0)
    with pytest.raises(ValueError):
        m.Fault(0.01, "load_ohm", 0)


def test_faults_actual_parameters():
    from hilconv.circuit import Circuit

    s = m.Schedule(Circuit(), [m.Fault(0.01, "load_ohm", 3), m.Fault(0.01, "input_v", 36)])
    assert s.at(0.009).input_v == 48
    assert s.at(0.01).input_v == 36 and s.at(0.01).load_ohm == 3
    assert s.inside(0, 0.02) == [0.01]
