import pytest

from hilconv import circuit as m


def test_circuit_passivity():
    assert m.Circuit().input_v == 48
    with pytest.raises(ValueError):
        m.Circuit(capacitor_esr_ohm=-1)
    with pytest.raises(ValueError):
        m.state([1, 2, 3])
