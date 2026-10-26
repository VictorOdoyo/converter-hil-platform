from hilconv import studies_analysis as m


def test_frequency():
    r = m.frequency()
    assert r["arrays"]["boost_phase_deg"][-1] < -200


def test_regulator():
    r = m.regulator()
    assert r["summary"]["spectral_radius"] < 1
    assert abs(r["summary"]["final_voltage_error_v"]) < 1e-5
