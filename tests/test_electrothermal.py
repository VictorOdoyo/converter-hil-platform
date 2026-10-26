import pytest

from hilconv import electrothermal as m


def test_device_parameters():
    assert m.Device().resistance_25_ohm == 0.02
    with pytest.raises(ValueError):
        m.Device(resistance_25_ohm=-1)
