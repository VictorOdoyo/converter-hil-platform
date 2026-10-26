import pytest

from hilconv import pwm as m


def test_topology_modes():
    assert m.mode("buck", 0) == (0, 1)
    assert m.mode("boost", 1) == (1, 0)
    assert m.mode("inverter", 0, 1) == (-1, 1)
    with pytest.raises(ValueError):
        m.mode("unknown", 1)
