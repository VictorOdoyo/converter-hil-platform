from hilconv import hybrid as m


def test_current_zero_event():
    from hilconv.circuit import Circuit, advance

    p = Circuit(inductor_ohm=0, capacitor_esr_ohm=0)
    t = m.first_current_crossing([1, 24], 50e-6, 0, 1, model=p)
    assert 8e-6 < t < 10e-6
    assert abs(advance([1, 24], t, 0, 1, p)["state"][0]) < 1e-9
