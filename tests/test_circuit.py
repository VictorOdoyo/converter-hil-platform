import pytest

from hilconv import circuit as m


def test_circuit_passivity():
    assert m.Circuit().input_v == 48
    with pytest.raises(ValueError):
        m.Circuit(capacitor_esr_ohm=-1)
    with pytest.raises(ValueError):
        m.state([1, 2, 3])


def test_kcl_esr():
    p = m.Circuit(capacitor_esr_ohm=0.2, load_source_v=10)
    r = m.ports([4, 20], 1, 1, p)
    assert 4 == pytest.approx(r["load_a"] + r["capacitor_a"])
    assert r["output_v"] - 20 == pytest.approx(0.2 * r["capacitor_a"])
