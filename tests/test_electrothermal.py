import pytest

from hilconv import electrothermal as m


def test_device_parameters():
    assert m.Device().resistance_25_ohm == 0.02
    with pytest.raises(ValueError):
        m.Device(resistance_25_ohm=-1)


def test_thermal_parameters():
    assert m.Thermal().total_resistance == pytest.approx(3.7)
    with pytest.raises(ValueError):
        m.Thermal(case_capacity_j_k=0)
