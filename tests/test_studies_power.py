import numpy as np
import pytest

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


def test_averaging():
    r = m.averaging()
    assert r["summary"]["fine_current_error_a"] < r["summary"]["coarse_current_error_a"] / 4


def test_esr():
    r = m.esr()
    assert r["summary"]["max_state_difference"] > 0.1
    assert r["arrays"]["correct_a"][0] == pytest.approx(r["arrays"]["naive_a"][0])


def test_inverter():
    r = m.inverter()
    assert 22 < r["summary"]["fundamental_peak_v"] < 25
    assert r["summary"]["low_order_thd"] < 0.05
    assert abs(r["summary"]["energy_residual_j"]) < 1e-8


def test_loss_estimate():
    r = m.loss_estimate()
    assert np.allclose(r["arrays"]["loss80_w"], 4 * r["arrays"]["loss20_w"])


def test_dcm_plot_shows_actual_conduction_pulses():
    r = m.dcm()
    assert max(r["arrays"]["pulse_current_a"]) > 0.5
    assert min(r["arrays"]["pulse_current_a"]) == 0
    assert any(p["x"] == "pulse_time_us" for p in r["plots"])
