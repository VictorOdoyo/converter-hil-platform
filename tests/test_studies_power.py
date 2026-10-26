from hilconv import studies_power as m


def test_buck():
    r = m.buck()
    assert abs(r["summary"]["current_ripple_a"] - 2.727272727) < 0.01
    assert r["summary"]["periodic_state_error"] < 1e-8
