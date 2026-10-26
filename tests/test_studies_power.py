from hilconv import studies_power as m


def test_buck():
    r = m.buck()
    assert abs(r["summary"]["current_ripple_a"] - 2.727272727) < 0.01
    assert r["summary"]["periodic_state_error"] < 1e-8


def test_boost():
    r = m.boost()
    assert r["summary"]["input_w"] > r["summary"]["output_w"] > 0
    assert r["summary"]["periodic_state_error"] < 1e-8


def test_regeneration():
    r = m.regeneration()
    assert r["summary"]["input_w"] < 0 and r["summary"]["output_w"] < 0
    assert r["summary"]["periodic_state_error"] < 1e-8


def test_bridge():
    r = m.bridge()
    assert r["summary"]["periodic_state_error"] < 1e-8
    assert r["summary"]["periodic_state_error"] < 1e-8


def test_dcm():
    r = m.dcm()
    assert r["summary"]["minimum_boundary_current_a"] >= 0
    assert r["summary"]["max_blocked_fraction"] > 0.3
    assert abs(r["summary"]["energy_residual_j"]) < 1e-9
