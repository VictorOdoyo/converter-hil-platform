from hilconv import studies_analysis as m


def test_frequency():
    r = m.frequency()
    assert r["arrays"]["boost_phase_deg"][-1] < -200
