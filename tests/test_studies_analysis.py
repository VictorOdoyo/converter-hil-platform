import pytest

from hilconv import studies_analysis as m


def test_frequency():
    r = m.frequency()
    assert r["arrays"]["boost_phase_deg"][-1] < -200


def test_regulator():
    r = m.regulator()
    assert r["summary"]["spectral_radius"] < 1
    assert abs(r["summary"]["final_voltage_error_v"]) < 1e-5


def test_observer():
    r = m.observer()
    assert r["summary"]["terminal_current_rmse_a"] < 0.1


def test_deadtime():
    r = m.deadtime()
    assert r["summary"]["peak_error_v"] == pytest.approx(1.92)


def test_deadline():
    r = m.deadline()
    assert r["summary"]["missed_deadlines"] > 6
